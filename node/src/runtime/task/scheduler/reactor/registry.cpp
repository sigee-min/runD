#include "registry.hpp"

#include "../../../reactor/readiness/mask.hpp"

#include "registration.hpp"

#include <algorithm>
#include <limits>

namespace rund::node {
namespace {

[[nodiscard]] auto FindOrder(ReactorRegistry &registry,
                             const std::uint64_t wait_id) noexcept {
  return std::lower_bound(
      registry.order.begin(), registry.order.end(), wait_id,
      [&registry](const std::uint32_t slot, const std::uint64_t value) {
        return registry.slots[slot].wait.wait_id < value;
      });
}

[[nodiscard]] auto FindOrder(const ReactorRegistry &registry,
                             const std::uint64_t wait_id) noexcept {
  return std::lower_bound(
      registry.order.begin(), registry.order.end(), wait_id,
      [&registry](const std::uint32_t slot, const std::uint64_t value) {
        return registry.slots[slot].wait.wait_id < value;
      });
}

[[nodiscard]] auto FindFd(ReactorRegistry &registry,
                          const ReactorHandle fd) noexcept {
  return std::lower_bound(
      registry.fds.begin(), registry.fds.end(), fd,
      [](const ReactorFdState &state, const ReactorHandle value) {
        return state.fd < value;
      });
}

[[nodiscard]] auto FindFd(const ReactorRegistry &registry,
                          const ReactorHandle fd) noexcept {
  return std::lower_bound(
      registry.fds.begin(), registry.fds.end(), fd,
      [](const ReactorFdState &state, const ReactorHandle value) {
        return state.fd < value;
      });
}

template <class Iterator>
[[nodiscard]] bool FdMatches(const Iterator found, const Iterator end,
                             const ReactorHandle fd) noexcept {
  return found != end && found->fd == fd;
}

[[nodiscard]] ReactorInterest Interest(const ReactorFdState &state) noexcept {
  ReactorInterest interest = ReactorInterest::None;
  if (state.read_count != 0u) {
    interest = interest | ReactorInterest::Read;
  }
  if (state.write_count != 0u) {
    interest = interest | ReactorInterest::Write;
  }
  return interest;
}

void AddInterest(ReactorFdState &state,
                 const ReactorInterest interest) noexcept {
  if (HasReactorInterest(interest, ReactorInterest::Read)) {
    ++state.read_count;
  }
  if (HasReactorInterest(interest, ReactorInterest::Write)) {
    ++state.write_count;
  }
}

[[nodiscard]] bool RemoveInterest(ReactorFdState &state,
                                  const ReactorInterest interest) noexcept {
  if ((HasReactorInterest(interest, ReactorInterest::Read) &&
       state.read_count == 0u) ||
      (HasReactorInterest(interest, ReactorInterest::Write) &&
       state.write_count == 0u)) {
    return false;
  }
  if (HasReactorInterest(interest, ReactorInterest::Read)) {
    --state.read_count;
  }
  if (HasReactorInterest(interest, ReactorInterest::Write)) {
    --state.write_count;
  }
  return true;
}

void InitializeFreeSlots(ReactorRegistry &registry) noexcept {
  registry.free_slots.clear();
  for (std::size_t index = registry.slots.size(); index != 0u; --index) {
    registry.slots[index - 1u] = ReactorWaitSlot{};
    registry.free_slots.push_back(static_cast<std::uint32_t>(index - 1u));
  }
  registry.live = 0u;
}

void ResetSlot(ReactorRegistry &registry, const std::uint32_t index) noexcept {
  ReactorWaitSlot &slot = registry.slots[index];
  slot.wait = ReactorWait{};
  slot.previous_fd = kNoReactorSlot;
  slot.next_fd = kNoReactorSlot;
}

void ReleaseSlot(ReactorRegistry &registry,
                 const std::uint32_t index) noexcept {
  ResetSlot(registry, index);
  registry.free_slots.push_back(index);
  --registry.live;
}

[[nodiscard]] bool Unlink(ReactorRegistry &registry,
                          ReactorFdState &fd,
                          const std::uint32_t index) noexcept {
  ReactorWaitSlot &slot = registry.slots[index];
  if (slot.wait.wait_id == 0u || slot.wait.fd != fd.fd ||
      fd.wait_count == 0u) {
    return false;
  }
  if (slot.previous_fd == kNoReactorSlot) {
    if (fd.first_wait != index) {
      return false;
    }
  } else if (slot.previous_fd >= registry.slots.size() ||
             registry.slots[slot.previous_fd].next_fd != index) {
    return false;
  }
  if (slot.next_fd == kNoReactorSlot) {
    if (fd.last_wait != index) {
      return false;
    }
  } else if (slot.next_fd >= registry.slots.size() ||
             registry.slots[slot.next_fd].previous_fd != index) {
    return false;
  }
  if (!RemoveInterest(fd, slot.wait.interest)) {
    return false;
  }
  if (slot.previous_fd == kNoReactorSlot) {
    fd.first_wait = slot.next_fd;
  } else {
    registry.slots[slot.previous_fd].next_fd = slot.next_fd;
  }
  if (slot.next_fd == kNoReactorSlot) {
    fd.last_wait = slot.previous_fd;
  } else {
    registry.slots[slot.next_fd].previous_fd = slot.previous_fd;
  }
  --fd.wait_count;
  slot.previous_fd = kNoReactorSlot;
  slot.next_fd = kNoReactorSlot;
  return true;
}

[[nodiscard]] bool ReadyMatches(const ReactorReady &ready,
                                const ReactorWait &wait) noexcept {
  return ready.wait_id == wait.wait_id && ready.task_id == wait.task_id &&
         ready.fd == wait.fd && ready.interest == wait.interest;
}

void ClearBatchMarks(
    ReactorRuntime &reactor,
    const std::vector<ReactorFdPreviousInterest> &affected) noexcept {
  for (const ReactorFdPreviousInterest &entry : affected) {
    ReactorFdState *const fd = ReactorRegistryFindFd(reactor, entry.fd);
    if (fd != nullptr) {
      fd->batch_touched = false;
    }
  }
}

} // namespace

bool ReactorRegistryPrepare(ReactorRuntime &reactor,
                            const std::size_t capacity) noexcept {
  if (capacity >= static_cast<std::size_t>(kNoReactorSlot) ||
      capacity > std::numeric_limits<std::size_t>::max() / 2u) {
    return false;
  }
  ReactorRegistry &registry = reactor.registry;
  try {
    registry.slots.assign(capacity, ReactorWaitSlot{});
    registry.order.clear();
    registry.order.reserve(capacity);
    registry.free_slots.clear();
    registry.free_slots.reserve(capacity);
    registry.fds.clear();
    registry.fds.reserve(capacity);
  } catch (...) {
    registry = ReactorRegistry{};
    return false;
  }
  InitializeFreeSlots(registry);
  return true;
}

void ReactorRegistryClear(ReactorRuntime &reactor) noexcept {
  ReactorRegistry &registry = reactor.registry;
  registry.order.clear();
  registry.fds.clear();
  registry.deferred_removes = 0u;
  InitializeFreeSlots(registry);
}

std::size_t ReactorRegistrySize(const ReactorRuntime &reactor) noexcept {
  return reactor.registry.live;
}

bool ReactorRegistryEmpty(const ReactorRuntime &reactor) noexcept {
  return reactor.registry.live == 0u;
}

const ReactorWait &
ReactorRegistryWaitAt(const ReactorRuntime &reactor,
                      const std::size_t index) noexcept {
  const ReactorRegistry &registry = reactor.registry;
  return registry.slots[registry.order[index]].wait;
}

const ReactorWait *
ReactorRegistryFindWait(const ReactorRuntime &reactor,
                        const std::uint64_t wait_id) noexcept {
  const ReactorRegistry &registry = reactor.registry;
  const auto found = FindOrder(registry, wait_id);
  if (found == registry.order.end()) {
    return nullptr;
  }
  const ReactorWaitSlot &slot = registry.slots[*found];
  return slot.wait.wait_id == wait_id ? &slot.wait : nullptr;
}

const ReactorFdState *
ReactorRegistryFindFd(const ReactorRuntime &reactor,
                      const ReactorHandle fd) noexcept {
  const ReactorRegistry &registry = reactor.registry;
  const auto found = FindFd(registry, fd);
  return FdMatches(found, registry.fds.end(), fd) ? &*found : nullptr;
}

ReactorFdState *ReactorRegistryFindFd(ReactorRuntime &reactor,
                                      const ReactorHandle fd) noexcept {
  ReactorRegistry &registry = reactor.registry;
  const auto found = FindFd(registry, fd);
  return FdMatches(found, registry.fds.end(), fd) ? &*found : nullptr;
}

bool ReactorRegistryEraseFd(ReactorRuntime &reactor,
                            const ReactorHandle fd) noexcept {
  ReactorRegistry &registry = reactor.registry;
  const auto found = FindFd(registry, fd);
  if (!FdMatches(found, registry.fds.end(), fd) || !found->erasable()) {
    return false;
  }
  registry.fds.erase(found);
  return true;
}

std::size_t ReactorRegistryFdCount(const ReactorRuntime &reactor) noexcept {
  return reactor.registry.fds.size();
}

std::uint32_t ReactorRegistryFirstWait(const ReactorRuntime &reactor,
                                       const ReactorHandle fd) noexcept {
  const ReactorFdState *const state = ReactorRegistryFindFd(reactor, fd);
  return state == nullptr ? kNoReactorSlot : state->first_wait;
}

std::uint32_t ReactorRegistryNextWait(const ReactorRuntime &reactor,
                                      const std::uint32_t slot) noexcept {
  return slot < reactor.registry.slots.size()
             ? reactor.registry.slots[slot].next_fd
             : kNoReactorSlot;
}

const ReactorWait *
ReactorRegistrySlotWait(const ReactorRuntime &reactor,
                        const std::uint32_t slot) noexcept {
  if (slot >= reactor.registry.slots.size() ||
      reactor.registry.slots[slot].wait.wait_id == 0u) {
    return nullptr;
  }
  return &reactor.registry.slots[slot].wait;
}

bool ReactorRegistryAddWait(ReactorRuntime &reactor,
                            const ReactorWait &wait) noexcept {
  ReactorRegistry &registry = reactor.registry;
  if (wait.wait_id == 0u) {
    return false;
  }
  auto order = registry.order.end();
  if (!registry.order.empty() &&
      registry.slots[registry.order.back()].wait.wait_id >= wait.wait_id) {
    order = FindOrder(registry, wait.wait_id);
  }
  if (order != registry.order.end() &&
      registry.slots[*order].wait.wait_id == wait.wait_id) {
    return false;
  }
  if (registry.free_slots.empty()) {
    return false;
  }

  auto fd = FindFd(registry, wait.fd);
  bool create_fd = !FdMatches(fd, registry.fds.end(), wait.fd);
  if (create_fd && registry.fds.size() == registry.fds.capacity()) {
    if (!ReactorRegistrationFlushDeferredRemoves(reactor)) {
      return false;
    }
    fd = FindFd(registry, wait.fd);
    create_fd = !FdMatches(fd, registry.fds.end(), wait.fd);
    if (create_fd && registry.fds.size() == registry.fds.capacity()) {
      return false;
    }
  }
  const std::uint32_t index = registry.free_slots.back();
  bool slot_taken = false;
  try {
    if (create_fd) {
      fd = registry.fds.insert(fd, ReactorFdState{.fd = wait.fd});
    }
    ReactorWaitSlot &slot = registry.slots[index];
    registry.free_slots.pop_back();
    slot_taken = true;
    slot.wait = wait;
    slot.previous_fd = fd->last_wait;
    slot.next_fd = kNoReactorSlot;
    if (!registry.order.empty() &&
        registry.slots[registry.order.back()].wait.wait_id >= wait.wait_id) {
      order = FindOrder(registry, wait.wait_id);
    } else {
      order = registry.order.end();
    }
    registry.order.insert(order, index);
    if (fd->last_wait == kNoReactorSlot) {
      fd->first_wait = index;
    } else {
      registry.slots[fd->last_wait].next_fd = index;
    }
    fd->last_wait = index;
    ++fd->wait_count;
    AddInterest(*fd, wait.interest);
    ++registry.live;
  } catch (...) {
    if (create_fd) {
      const auto rollback = FindFd(registry, wait.fd);
      if (FdMatches(rollback, registry.fds.end(), wait.fd) &&
          rollback->wait_count == 0u) {
        registry.fds.erase(rollback);
      }
    }
    if (slot_taken) {
      ResetSlot(registry, index);
      registry.free_slots.push_back(index);
    }
    return false;
  }
  return true;
}

bool ReactorRegistryRemoveWait(
    ReactorRuntime &reactor, const std::uint64_t wait_id,
    ReactorWait *const removed,
    ReactorInterest *const previous_interest) noexcept {
  if (previous_interest != nullptr) {
    *previous_interest = ReactorInterest::None;
  }
  ReactorRegistry &registry = reactor.registry;
  const auto found = FindOrder(registry, wait_id);
  if (found == registry.order.end()) {
    return false;
  }
  const std::uint32_t index = *found;
  ReactorWaitSlot &slot = registry.slots[index];
  if (slot.wait.wait_id != wait_id) {
    return false;
  }
  ReactorFdState *fd = ReactorRegistryFindFd(reactor, slot.wait.fd);
  if (fd == nullptr) {
    return false;
  }
  if (previous_interest != nullptr) {
    *previous_interest = Interest(*fd);
  }
  if (removed != nullptr) {
    *removed = slot.wait;
  }
  if (!Unlink(registry, *fd, index)) {
    return false;
  }
  const ReactorHandle removed_fd = slot.wait.fd;
  registry.order.erase(found);
  ReleaseSlot(registry, index);
  fd = ReactorRegistryFindFd(reactor, removed_fd);
  if (fd != nullptr && fd->erasable()) {
    static_cast<void>(ReactorRegistryEraseFd(reactor, removed_fd));
  }
  return true;
}

bool ReactorRegistryRemoveReadyBatch(
    ReactorRuntime &reactor, const std::vector<ReactorReady> &ordered,
    std::vector<ReactorWait> &removed,
    std::vector<ReactorFdPreviousInterest> &affected) noexcept {
  ReactorRegistry &registry = reactor.registry;
  const std::size_t free_start = registry.free_slots.size();
  try {
    removed.clear();
    removed.reserve(ordered.size());
    affected.clear();
    affected.reserve(ordered.size());
    for (const ReactorReady &ready : ordered) {
      const auto found = FindOrder(registry, ready.wait_id);
      if (found == registry.order.end()) {
        break;
      }
      const ReactorWaitSlot &slot = registry.slots[*found];
      if (!ReadyMatches(ready, slot.wait) ||
          (!removed.empty() && removed.back().wait_id == ready.wait_id)) {
        break;
      }
      removed.push_back(slot.wait);
      registry.free_slots.push_back(*found);
    }
  } catch (...) {
    removed.clear();
    affected.clear();
    registry.free_slots.resize(free_start);
    return false;
  }

  const bool removed_all = removed.size() == ordered.size();
  for (std::size_t free_index = free_start;
       free_index < registry.free_slots.size(); ++free_index) {
    const std::uint32_t index = registry.free_slots[free_index];
    ReactorWaitSlot &slot = registry.slots[index];
    ReactorFdState *const fd = ReactorRegistryFindFd(reactor, slot.wait.fd);
    if (fd == nullptr) {
      ClearBatchMarks(reactor, affected);
      removed.clear();
      affected.clear();
      registry.free_slots.resize(free_start);
      return false;
    }
    if (!fd->batch_touched) {
      fd->batch_touched = true;
      affected.push_back(
          ReactorFdPreviousInterest{.fd = fd->fd,
                                    .interest = Interest(*fd)});
    }
    if (!Unlink(registry, *fd, index)) {
      ClearBatchMarks(reactor, affected);
      removed.clear();
      affected.clear();
      registry.free_slots.resize(free_start);
      return false;
    }
    ResetSlot(registry, index);
    --registry.live;
  }
  registry.order.erase(
      std::remove_if(registry.order.begin(), registry.order.end(),
                     [&registry](const std::uint32_t index) {
                       return registry.slots[index].wait.wait_id == 0u;
                     }),
      registry.order.end());
  ClearBatchMarks(reactor, affected);
  return removed_all;
}

ReactorInterest ReactorRegistryInterestForFd(const ReactorRuntime &reactor,
                                             const ReactorHandle fd) noexcept {
  const ReactorFdState *const state = ReactorRegistryFindFd(reactor, fd);
  return state == nullptr ? ReactorInterest::None : Interest(*state);
}

bool ReactorRegistryCollectChangesForWaitAdd(ReactorRuntime &reactor,
                                             const ReactorWait &wait) noexcept {
  const ReactorInterest current =
      ReactorRegistryInterestForFd(reactor, wait.fd);
  return ReactorRegistrationCollectForWaitAdd(reactor, wait.fd, current,
                                              wait.fd_generation);
}

bool ReactorRegistryCollectChangesForWaitRemove(
    ReactorRuntime &reactor, const ReactorHandle fd,
    const ReactorInterest previous_interest) noexcept {
  const ReactorInterest current = ReactorRegistryInterestForFd(reactor, fd);
  return ReactorRegistrationCollectForWaitRemove(reactor, fd, previous_interest,
                                                 current);
}

} // namespace rund::node

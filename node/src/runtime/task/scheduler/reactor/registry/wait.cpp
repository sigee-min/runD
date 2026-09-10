#include "local.hpp"

#include "../registration.hpp"

namespace rund::node {

using reactor_registry_detail::AddInterest;
using reactor_registry_detail::FdMatches;
using reactor_registry_detail::FindFd;
using reactor_registry_detail::FindOrder;
using reactor_registry_detail::Interest;
using reactor_registry_detail::ReleaseSlot;
using reactor_registry_detail::ResetSlot;
using reactor_registry_detail::Unlink;

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

} // namespace rund::node

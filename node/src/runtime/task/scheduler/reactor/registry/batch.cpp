#include "local.hpp"

#include <algorithm>

namespace rund::node {
namespace {

using reactor_registry_detail::FindOrder;
using reactor_registry_detail::Interest;
using reactor_registry_detail::ResetSlot;
using reactor_registry_detail::Unlink;

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
          ReactorFdPreviousInterest{.fd = fd->fd, .interest = Interest(*fd)});
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

} // namespace rund::node

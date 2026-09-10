#include "local.hpp"

#include "../registration.hpp"

namespace rund::node {

using reactor_registry_detail::FdMatches;
using reactor_registry_detail::FindFd;
using reactor_registry_detail::FindOrder;
using reactor_registry_detail::Interest;

const ReactorWait &ReactorRegistryWaitAt(const ReactorRuntime &reactor,
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

const ReactorFdState *ReactorRegistryFindFd(const ReactorRuntime &reactor,
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

const ReactorWait *ReactorRegistrySlotWait(const ReactorRuntime &reactor,
                                           const std::uint32_t slot) noexcept {
  if (slot >= reactor.registry.slots.size() ||
      reactor.registry.slots[slot].wait.wait_id == 0u) {
    return nullptr;
  }
  return &reactor.registry.slots[slot].wait;
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

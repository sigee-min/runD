#include "registry.hpp"

#include <limits>

namespace rund::node {
namespace {

void InitializeFreeSlots(ReactorRegistry &registry) noexcept {
  registry.free_slots.clear();
  for (std::size_t index = registry.slots.size(); index != 0u; --index) {
    registry.slots[index - 1u] = ReactorWaitSlot{};
    registry.free_slots.push_back(static_cast<std::uint32_t>(index - 1u));
  }
  registry.live = 0u;
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

} // namespace rund::node

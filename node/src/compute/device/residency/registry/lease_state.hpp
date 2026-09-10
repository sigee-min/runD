#pragma once

#include "../execution/graph_persist/capacity.hpp"
#include "model/lease.hpp"

#include <algorithm>
#include <array>

namespace rund::compute::detail::residency {

[[nodiscard]] inline bool
active(const registry_model::LeaseSlot &slot) noexcept {
  return slot.state != registry_model::LeaseState::Free;
}

template <std::size_t N>
[[nodiscard]] inline bool
any_active(const std::array<registry_model::LeaseSlot, N> &slots) noexcept {
  return std::any_of(
      slots.begin(), slots.end(),
      [](const registry_model::LeaseSlot &slot) { return active(slot); });
}

[[nodiscard]] inline bool retry_ready(
    const std::array<registry_model::LeaseSlot,
                     execution::GraphPersistSlotCapacity> &slots) noexcept {
  return std::any_of(
      slots.begin(), slots.end(), [](const registry_model::LeaseSlot &slot) {
        return slot.state == registry_model::LeaseState::RetryReady;
      });
}

} // namespace rund::compute::detail::residency

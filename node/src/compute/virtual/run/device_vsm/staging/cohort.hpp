#pragma once

#include "../../../backing.hpp"
#include "../../../state.hpp"

#include <cstdint>
#include <memory>
#include <span>
#include <utility>

namespace rund::compute::detail::device_vsm_product_detail {

[[nodiscard]] inline std::shared_ptr<VirtualReadCohort>
graph_read_cohort(const std::span<VirtualBacking *const> inputs) noexcept {
  if (inputs.empty() ||
      inputs.size() > VirtualPipelineState::InputCapacity) {
    return {};
  }
  std::shared_ptr<VirtualReadCohort> provider;
  VirtualCohortId cohort{};
  std::uint32_t lanes = 0u;
  for (VirtualBacking *const input : inputs) {
    if (input == nullptr || VirtualBackingAccess::resident(*input) != nullptr) {
      return {};
    }
    auto *const member = dynamic_cast<VirtualBackingReadCohort *>(input);
    if (member == nullptr) {
      return {};
    }
    std::shared_ptr<VirtualReadCohort> current = member->cohort();
    if (current == nullptr) {
      return {};
    }
    const VirtualCohortId current_id = current->cohort_id();
    const std::uint32_t current_lanes = current->lane_limit();
    if (!current_id || current_lanes < 1u || current_lanes > 2u) {
      return {};
    }
    if (provider == nullptr) {
      provider = std::move(current);
      cohort = current_id;
      lanes = current_lanes;
    } else if (current.get() != provider.get() || current_id != cohort ||
               current_lanes != lanes) {
      return {};
    }
  }
  return provider;
}

} // namespace rund::compute::detail::device_vsm_product_detail

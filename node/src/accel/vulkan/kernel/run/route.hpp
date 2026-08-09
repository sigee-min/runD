#pragma once

#include "../../../kernel/backend/run.hpp"
#include "../../../resident/window/admission/runtime/windows.hpp"

#include <cstddef>
#include <cstdint>

namespace rund::node::accel::detail {

[[nodiscard]] inline std::uint64_t
VulkanMapRouteDispatches(const rund::kernel::ComputePlan &plan,
                         const BoundStep *const bound) noexcept {
  if (bound == nullptr || bound->planned == nullptr ||
      bound->planned->windows.size() == 0u) {
    return plan.dispatch_count;
  }
  return RuntimeWindowCount(plan, MapBindingFor(*bound),
                            bound->planned->windows.size());
}

[[nodiscard]] inline std::uint64_t UniqueVulkanMapCheckCount(
    const rund::kernel::LoweringArtifact &artifact) noexcept {
  std::uint64_t count = 0u;
  for (std::size_t index = 0u; index < artifact.metadata.read_routes.size();
       ++index) {
    bool first = true;
    for (std::size_t prior = 0u; prior < index; ++prior) {
      if (artifact.metadata.read_routes[prior].index ==
          artifact.metadata.read_routes[index].index) {
        first = false;
        break;
      }
    }
    count += first ? 1u : 0u;
  }
  return count;
}

} // namespace rund::node::accel::detail

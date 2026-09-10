#include "../planner.hpp"

#include "../identity.hpp"

#include "../../../type.hpp"
#include "../../plan/contract.hpp"

#include <kernel/core/checked.hpp>

#include <algorithm>
#include <cstdint>
#include <limits>

namespace rund::compute::detail::residency {

PlanResult PlanResidency(const StreamPlanInput &input) noexcept {
  std::uint64_t dirty_capacity = 0u;
  std::uint64_t last_dirty_offset = 0u;
  if (input.page_bytes == 0u || input.dirty.bytes == 0u ||
      input.dirty.offset >= input.page_bytes ||
      input.dirty.bytes > input.page_bytes - input.dirty.offset ||
      !kernel::checked::mul(input.page_count, input.dirty.bytes,
                            dirty_capacity) ||
      (input.page_count != 0u &&
       (!kernel::checked::mul(input.page_count - 1u, input.dirty.bytes,
                              last_dirty_offset) ||
        input.dirty_bytes <= last_dirty_offset ||
        input.dirty_bytes > dirty_capacity)) ||
      (input.page_count == 0u && input.dirty_bytes != 0u)) {
    return PlanResult{.failure = Failure::Invalid};
  }
  if (input.requested_frames > input.max_frames ||
      input.max_frames > std::numeric_limits<std::uint32_t>::max()) {
    return PlanResult{.failure = Failure::Capacity};
  }
  if (input.page_count != 0u && input.requested_frames == 0u) {
    return PlanResult{.failure = Failure::Infeasible};
  }
  const std::uint64_t frames =
      std::min(input.page_count, input.requested_frames);
  const StreamPlan stream{input.page_count, frames, input.dirty,
                          input.dirty_bytes, input.prefetch_distance};
  const Identity identity = IdentifyResidencyPlan(
      input.page_bytes, input.page_count, frames, input.dirty,
      input.dirty_bytes, input.prefetch_distance);
  return PlanResult{
      .failure = Failure::None,
      .plan = ResidencyPlan{input.page_bytes, stream, identity},
  };
}

} // namespace rund::compute::detail::residency

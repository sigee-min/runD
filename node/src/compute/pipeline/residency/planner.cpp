#include "planner.hpp"

#include "identity.hpp"

#include <algorithm>
#include <cstdint>

namespace rund::compute::detail::residency {

PlanResult PlanResidency(const PlanInput &input) noexcept {
  if (input.page_bytes == 0u) {
    return PlanResult{.failure = Failure::Invalid};
  }
  if (input.requested_slots > input.max_slots) {
    return PlanResult{.failure = Failure::Capacity};
  }
  if (input.page_count != 0u && input.requested_slots == 0u) {
    return PlanResult{.failure = Failure::Infeasible};
  }
  const std::uint64_t slots = std::min(input.page_count, input.requested_slots);
  const LinearPlan linear{input.page_count, slots};
  const Identity identity =
      IdentifyResidencyPlan(input.page_bytes, input.page_count, slots);
  return PlanResult{
      .failure = Failure::None,
      .plan = ResidencyPlan{input.page_bytes, linear, identity},
  };
}

} // namespace rund::compute::detail::residency

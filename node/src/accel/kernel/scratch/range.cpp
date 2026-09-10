#include "../scratch.hpp"

#include "../../range_aggregate/model/plan.hpp"

#include <array>
#include <limits>

namespace rund::node::accel::detail {
namespace {

inline constexpr std::uint32_t kRangeScratchRoleTag = 0x52410000u;

[[nodiscard]] constexpr std::uint32_t
range_role_value(const RangeTempRole role) noexcept {
  switch (role) {
  case RangeTempRole::PrefixValues:
    return 0u;
  case RangeTempRole::BlockSummaries:
    return 1u;
  case RangeTempRole::ForwardValues:
    return 2u;
  case RangeTempRole::BackwardValues:
    return 3u;
  }
  return std::numeric_limits<std::uint32_t>::max();
}

} // namespace

KernelScratchRole ScratchRoleForRangeTemp(const RangeTempRole role,
                                          const std::uint8_t ordinal) noexcept {
  const std::uint32_t value = range_role_value(role);
  return value == std::numeric_limits<std::uint32_t>::max()
             ? KernelScratchRole{}
             : KernelScratchRole::from(kRangeScratchRoleTag | (value << 8u) |
                                       ordinal);
}

KernelScratchRequirement
ScratchReqForRangeTemp(const RangeTempReq &requirement) noexcept {
  return KernelScratchRequirement{
      .role = ScratchRoleForRangeTemp(requirement.role, requirement.ordinal),
      .bytes = requirement.bytes,
      .alignment = requirement.alignment,
      .first_stage = requirement.first_stage,
      .last_stage = requirement.last_stage,
  };
}

KernelScratchBatchPlan PlanRangeScratch(const RangePlan &plan,
                                        const std::uint64_t backing_alignment,
                                        const std::uint64_t page_bytes) {
  if (!plan.ok()) {
    return KernelScratchBatchPlan::failure(plan.reason());
  }
  const std::size_t stage_count = plan.stage_count();
  const std::size_t temporary_count = plan.temporary_count();
  if (stage_count == 0u || stage_count > kRangeStageCap ||
      temporary_count > kRangeTempCap) {
    return KernelScratchBatchPlan::failure("accel_kernel_scratch_invalid");
  }
  std::array<KernelScratchRequirement, kRangeTempCap> requirements{};
  for (std::size_t index = 0u; index < temporary_count; ++index) {
    const RangeTempReq temporary = plan.temporary(index);
    requirements[index] = ScratchReqForRangeTemp(temporary);
    if (!requirements[index].valid() ||
        requirements[index].last_stage >= stage_count) {
      return KernelScratchBatchPlan::failure("accel_kernel_scratch_invalid");
    }
  }
  return PlanKernelScratchBatch(
      std::span<const KernelScratchRequirement>{requirements.data(),
                                                temporary_count},
      backing_alignment, page_bytes);
}

const KernelScratchPlacement *
FindRangeScratch(const KernelScratchBatchPlan &plan, const RangeTempRole role,
                 const std::uint8_t ordinal) noexcept {
  return FindKernelScratchPlacement(plan,
                                    ScratchRoleForRangeTemp(role, ordinal));
}

} // namespace rund::node::accel::detail

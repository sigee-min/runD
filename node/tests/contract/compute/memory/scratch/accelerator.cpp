#include "../local.hpp"

#include "../../../../../src/accel/kernel/scratch.hpp"
#include "../../../../../src/accel/range_aggregate/plan.hpp"

#include <array>
#include <cstdint>
#include <limits>
#include <span>
#include <string_view>

namespace rund_node_memory_contract {

[[nodiscard]] int CheckAcceleratorScratchPlacementAuthority() {
  using namespace rund::node::accel::detail;
  constexpr KernelScratchRole prefix = KernelScratchRole::from(0u);
  constexpr KernelScratchRole summary = KernelScratchRole::from(1u);
  constexpr KernelScratchRole suffix = KernelScratchRole::from(2u);
  constexpr KernelScratchRole fixup = KernelScratchRole::from(3u);
  constexpr std::array<KernelScratchRequirement, 4u> requirements{
      KernelScratchRequirement{
          .role = prefix,
          .bytes = 24u,
          .alignment = 8u,
          .first_stage = 0u,
          .last_stage = 1u,
      },
      KernelScratchRequirement{
          .role = summary,
          .bytes = 16u,
          .alignment = 16u,
          .first_stage = 0u,
          .last_stage = 0u,
      },
      KernelScratchRequirement{
          .role = suffix,
          .bytes = 32u,
          .alignment = 8u,
          .first_stage = 1u,
          .last_stage = 2u,
      },
      KernelScratchRequirement{
          .role = fixup,
          .bytes = 8u,
          .alignment = 8u,
          .first_stage = 2u,
          .last_stage = 3u,
      },
  };
  const KernelScratchBatchPlan planned =
      PlanKernelScratchBatch(requirements, 16u, 64u);
  const KernelScratchPlacement *const prefix_place =
      FindKernelScratchPlacement(planned, prefix);
  const KernelScratchPlacement *const summary_place =
      FindKernelScratchPlacement(planned, summary);
  const KernelScratchPlacement *const suffix_place =
      FindKernelScratchPlacement(planned, suffix);
  const KernelScratchPlacement *const fixup_place =
      FindKernelScratchPlacement(planned, fixup);
  KernelScratchPlacement misaligned =
      prefix_place == nullptr ? KernelScratchPlacement{} : *prefix_place;
  misaligned.offset = 8u;
  if (!planned.ok() || std::string_view{planned.reason()} != "ok" ||
      planned.page_count() != 1u || planned.last_bytes() != 64u ||
      planned.backing_bytes() != 64u || planned.payload_bytes() != 56u ||
      planned.placements().size() != requirements.size() ||
      prefix_place == nullptr || prefix_place->page != 0u ||
      prefix_place->offset != 0u || summary_place == nullptr ||
      summary_place->page != 0u || summary_place->offset != 32u ||
      suffix_place == nullptr || suffix_place->page != 0u ||
      suffix_place->offset != 32u || fixup_place == nullptr ||
      fixup_place->page != 0u || fixup_place->offset != 0u ||
      !scratch::valid(*prefix_place, 16u, 64u) ||
      scratch::valid(misaligned, 16u, 64u) ||
      FindKernelScratchPlacement(planned, KernelScratchRole::from(99u)) !=
          nullptr) {
    return 1;
  }

  // Input order is not a second placement authority. Canonical stage/role
  // order produces the exact same frozen placement vector.
  constexpr std::array<KernelScratchRequirement, 4u> permuted{
      requirements[3u], requirements[1u], requirements[0u], requirements[2u]};
  const KernelScratchBatchPlan repeated =
      PlanKernelScratchBatch(permuted, 16u, 64u);
  if (!repeated.ok() || repeated.placements() != planned.placements() ||
      repeated.payload_bytes() != planned.payload_bytes() ||
      repeated.backing_bytes() != planned.backing_bytes() ||
      repeated.last_bytes() != planned.last_bytes() ||
      repeated.page_count() != planned.page_count()) {
    return 2;
  }

  constexpr std::array<KernelScratchRequirement, 2u> concurrent{
      KernelScratchRequirement{
          .role = prefix,
          .bytes = 48u,
          .alignment = 16u,
          .first_stage = 0u,
          .last_stage = 0u,
      },
      KernelScratchRequirement{
          .role = summary,
          .bytes = 32u,
          .alignment = 16u,
          .first_stage = 0u,
          .last_stage = 0u,
      },
  };
  const KernelScratchBatchPlan paged =
      PlanKernelScratchBatch(concurrent, 16u, 64u);
  if (!paged.ok() || paged.page_count() != 2u || paged.last_bytes() != 32u ||
      paged.backing_bytes() != 96u || paged.payload_bytes() != 80u) {
    return 3;
  }
  std::array<KernelScratchRequirement, 2u> serial = concurrent;
  serial[1u].first_stage = 1u;
  serial[1u].last_stage = 1u;
  const KernelScratchBatchPlan reused =
      PlanKernelScratchBatch(serial, 16u, 64u);
  const KernelScratchPlacement *const serial_first =
      FindKernelScratchPlacement(reused, prefix);
  const KernelScratchPlacement *const serial_second =
      FindKernelScratchPlacement(reused, summary);
  if (!reused.ok() || reused.page_count() != 1u || reused.last_bytes() != 48u ||
      reused.backing_bytes() != 48u || reused.payload_bytes() != 48u ||
      serial_first == nullptr || serial_second == nullptr ||
      serial_first->page != serial_second->page ||
      serial_first->offset != serial_second->offset) {
    return 4;
  }

  const KernelScratchBatchPlan empty = PlanKernelScratchBatch({}, 16u, 64u);
  KernelScratchRequirement invalid = requirements[0u];
  invalid.role = {};
  const KernelScratchBatchPlan invalid_role =
      PlanKernelScratchBatch(std::span{&invalid, 1u}, 16u, 64u);
  invalid = requirements[0u];
  invalid.first_stage = 2u;
  invalid.last_stage = 1u;
  const KernelScratchBatchPlan invalid_lifetime =
      PlanKernelScratchBatch(std::span{&invalid, 1u}, 16u, 64u);
  invalid = requirements[0u];
  invalid.alignment = 32u;
  const KernelScratchBatchPlan invalid_alignment =
      PlanKernelScratchBatch(std::span{&invalid, 1u}, 16u, 64u);
  invalid = requirements[0u];
  invalid.alignment = 3u;
  const KernelScratchBatchPlan invalid_power =
      PlanKernelScratchBatch(std::span{&invalid, 1u}, 16u, 64u);
  invalid = requirements[0u];
  invalid.bytes = 0u;
  const KernelScratchBatchPlan invalid_bytes =
      PlanKernelScratchBatch(std::span{&invalid, 1u}, 16u, 64u);
  invalid = requirements[0u];
  invalid.bytes = 65u;
  const KernelScratchBatchPlan oversized =
      PlanKernelScratchBatch(std::span{&invalid, 1u}, 16u, 64u);
  std::array<KernelScratchRequirement, 2u> duplicate{requirements[0u],
                                                     requirements[0u]};
  const KernelScratchBatchPlan duplicate_role =
      PlanKernelScratchBatch(duplicate, 16u, 64u);
  if (!empty.ok() || empty.page_count() != 0u || empty.payload_bytes() != 0u ||
      empty.backing_bytes() != 0u || invalid_role.ok() ||
      invalid_lifetime.ok() || invalid_alignment.ok() || invalid_power.ok() ||
      invalid_bytes.ok() || duplicate_role.ok() || oversized.ok() ||
      std::string_view{oversized.reason()} !=
          "compute_resident_bytes_invalid" ||
      PlanKernelScratchBatch(requirements, 0u, 64u).ok() ||
      PlanKernelScratchBatch(requirements, 16u, 63u).ok()) {
    return 5;
  }

  constexpr std::array<KernelScratchRequirement, 2u> overflow{
      KernelScratchRequirement{
          .role = prefix,
          .bytes = std::numeric_limits<std::uint64_t>::max(),
          .alignment = 1u,
          .first_stage = 0u,
          .last_stage = 0u,
      },
      KernelScratchRequirement{
          .role = summary,
          .bytes = 1u,
          .alignment = 1u,
          .first_stage = 0u,
          .last_stage = 0u,
      },
  };
  const KernelScratchBatchPlan overflowed = PlanKernelScratchBatch(
      overflow, 1u, std::numeric_limits<std::uint64_t>::max());
  if (overflowed.ok() ||
      std::string_view{overflowed.reason()} != "compute_pipeline_capacity") {
    return 6;
  }

  constexpr std::uint8_t direct_prefix =
      RangeSupportBit(RangeSupport::Direct) |
      RangeSupportBit(RangeSupport::PrefixDifference);
  const auto sum_traits =
      RangeTraits::sum_modulo(rund::kernel::ComputeDomain::U32);
  const auto sum_shape =
      sum_traits.has_value()
          ? RangeShape::affine(*sum_traits, RangeBoundary::Clamp, 4097u, 4097u,
                               8195u, 1u, 4097u, 4u)
          : std::nullopt;
  const auto prefix_caps =
      RangeCaps::gpu(RangeSource::Metal, kRangeWidth64Bit, 64u, 4u, 32768u,
                     std::numeric_limits<std::uint32_t>::max(),
                     std::numeric_limits<std::uint64_t>::max(), direct_prefix);
  if (!sum_shape.has_value() || !prefix_caps.has_value()) {
    return 7;
  }
  const RangePlan prefix_plan = PlanRange(*sum_shape, *prefix_caps);
  const KernelScratchBatchPlan prefix_scratch =
      PlanRangeScratch(prefix_plan, 16u, 32768u);
  const KernelScratchBatchPlan prefix_repeated =
      PlanRangeScratch(prefix_plan, 16u, 32768u);
  if (!prefix_plan.ok() ||
      prefix_plan.candidate().disposition() != RangePath::PrefixDifference ||
      !prefix_scratch.ok() || prefix_scratch.page_count() != 1u ||
      prefix_scratch.last_bytes() != 16688u ||
      prefix_scratch.backing_bytes() != 16688u ||
      prefix_scratch.payload_bytes() != 16656u ||
      prefix_scratch.placements() != prefix_repeated.placements() ||
      prefix_scratch.payload_bytes() != prefix_repeated.payload_bytes() ||
      prefix_scratch.backing_bytes() != prefix_repeated.backing_bytes()) {
    return 8;
  }
  for (std::size_t index = 0u; index < prefix_plan.temporary_count(); ++index) {
    const RangeTempReq temporary = prefix_plan.temporary(index);
    const KernelScratchRequirement projected =
        ScratchReqForRangeTemp(temporary);
    const KernelScratchPlacement *const placed =
        FindRangeScratch(prefix_scratch, temporary.role, temporary.ordinal);
    if (!projected.valid() || placed == nullptr ||
        placed->requirement != projected) {
      return 9;
    }
  }
  if (ScratchRoleForRangeTemp(static_cast<RangeTempRole>(255u), 0u).valid() ||
      ScratchRoleForRangeTemp(RangeTempRole::PrefixValues, 0u) ==
          ScratchRoleForRangeTemp(RangeTempRole::BlockSummaries, 0u) ||
      ScratchRoleForRangeTemp(RangeTempRole::PrefixValues, 0u) ==
          ScratchRoleForRangeTemp(RangeTempRole::PrefixValues, 1u)) {
    return 10;
  }

  constexpr std::uint8_t direct_block =
      RangeSupportBit(RangeSupport::Direct) |
      RangeSupportBit(RangeSupport::BlockPrefixSuffix);
  const auto minimum_traits =
      RangeTraits::minimum(rund::kernel::ComputeDomain::U32);
  const auto minimum_shape =
      minimum_traits.has_value()
          ? RangeShape::affine(*minimum_traits, RangeBoundary::Clamp, 4097u,
                               4097u, 8195u, 1u, 4097u, 4u)
          : std::nullopt;
  const auto block_caps =
      RangeCaps::gpu(RangeSource::Vulkan, kRangeWidth64Bit, 64u, 0u, 0u,
                     std::numeric_limits<std::uint32_t>::max(),
                     std::numeric_limits<std::uint32_t>::max(), direct_block);
  if (!minimum_shape.has_value() || !block_caps.has_value()) {
    return 11;
  }
  const RangePlan block_plan = PlanRange(*minimum_shape, *block_caps);
  const KernelScratchBatchPlan block_scratch =
      PlanRangeScratch(block_plan, 16u, 65536u);
  const KernelScratchPlacement *const forward =
      FindRangeScratch(block_scratch, RangeTempRole::ForwardValues, 0u);
  const KernelScratchPlacement *const backward =
      FindRangeScratch(block_scratch, RangeTempRole::BackwardValues, 0u);
  if (!block_plan.ok() ||
      block_plan.candidate().disposition() != RangePath::BlockPrefixSuffix ||
      !block_scratch.ok() || block_scratch.page_count() != 2u ||
      block_scratch.last_bytes() != 49168u ||
      block_scratch.backing_bytes() != 114704u ||
      block_scratch.payload_bytes() != 98328u || forward == nullptr ||
      backward == nullptr || forward->requirement.first_stage != 0u ||
      forward->requirement.last_stage != 1u ||
      backward->requirement.first_stage != 0u ||
      backward->requirement.last_stage != 1u ||
      forward->requirement.role == backward->requirement.role) {
    return 12;
  }

  const RangePlan direct_plan =
      PlanRange(*sum_shape, RangeCaps::cpu_reference());
  const KernelScratchBatchPlan direct_scratch =
      PlanRangeScratch(direct_plan, 16u, 32768u);
  const KernelScratchBatchPlan rejected_scratch =
      PlanRangeScratch(RangePlan::rejected("range_rejected"), 16u, 32768u);
  if (!direct_scratch.ok() || direct_scratch.page_count() != 0u ||
      direct_scratch.payload_bytes() != 0u || rejected_scratch.ok() ||
      std::string_view{rejected_scratch.reason()} != "range_rejected") {
    return 13;
  }
  return 0;
}

} // namespace rund_node_memory_contract

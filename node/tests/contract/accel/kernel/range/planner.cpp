#include "local.hpp"

#include "src/accel/context/internal/execution.hpp"
#include "src/accel/metal/range/local.hpp"
#include "src/accel/source/hash.hpp"
#include "src/accel/vulkan/kernel/manifest.hpp"
#include "src/accel/vulkan/kernel/ops/table.hpp"
#include "src/accel/vulkan/kernel/pipeline/source.hpp"
#include "src/accel/vulkan/kernel/pipeline/template.hpp"
#include "src/accel/vulkan/range/local.hpp"
#include "src/accel/window/shape.hpp"

#include <kernel/program/compute/window/plan.hpp>

#include <array>
#include <cstdint>
#include <limits>
#include <optional>
#include <string>
#include <string_view>

namespace node_accel_contract::range {
namespace {

using rund::kernel::ComputeDomain;
using rund::kernel::u32;
using rund::kernel::u64;
using namespace rund::node::accel::detail;

[[nodiscard]] constexpr bool BasicSelectionContract() {
  const RangeCaps full = Gpu(RangeSource::Metal);
  const RangePlan small_sum = PlanRange(Shape(RangeOp::Sum, 515u, 1u), full);
  const RangePlan large_sum = PlanRange(Shape(RangeOp::Sum, 515u, 515u), full);
  const RangePlan small_min =
      PlanRange(Shape(RangeOp::Minimum, 515u, 1u), full);
  const RangePlan large_max =
      PlanRange(Shape(RangeOp::Maximum, 515u, 515u), full);
  const RangePlan cpu =
      PlanRange(Shape(RangeOp::Sum, 17u, 3u), RangeCaps::cpu_reference());
  return small_sum.ok() &&
         small_sum.candidate().disposition() == RangePath::SharedHalo &&
         small_sum.candidate().width() == 256u && large_sum.ok() &&
         large_sum.candidate().disposition() == RangePath::PrefixDifference &&
         small_min.ok() &&
         small_min.candidate().disposition() == RangePath::SharedHalo &&
         large_max.ok() &&
         large_max.candidate().disposition() == RangePath::BlockPrefixSuffix &&
         cpu.ok() && cpu.candidate() == RangeCandidate::direct_cpu() &&
         cpu.stage_count() == 1u && cpu.temporary_count() == 0u;
}

[[nodiscard]] constexpr bool WidthAndCapacityContract() {
  constexpr std::array<u32, 3u> widths{64u, 128u, 256u};
  constexpr std::array<std::uint8_t, 3u> width_bits{
      kRangeWidth64Bit, kRangeWidth128Bit, kRangeWidth256Bit};
  constexpr std::uint8_t direct_shared =
      RangeSupportBit(RangeSupport::Direct) |
      RangeSupportBit(RangeSupport::SharedHalo);
  for (std::size_t width_index = 0u; width_index < widths.size();
       ++width_index) {
    const u32 width = widths[width_index];
    const std::array<u32, 3u> capacities{9u, width - 1u, width};
    for (const u32 capacity : capacities) {
      for (const u32 element_bytes : std::array<u32, 2u>{4u, 8u}) {
        const u64 exact_limit =
            static_cast<u64>(width + 2u * capacity) * element_bytes * 4u;
        const RangeShape shape = Shape(
            RangeOp::Sum, width + 1u, capacity, element_bytes,
            element_bytes == 4u ? ComputeDomain::U32 : ComputeDomain::U64);
        const RangePlan below = PlanRange(
            shape, Gpu(RangeSource::Metal, width_bits[width_index], width, 4u,
                       exact_limit - 1u, std::numeric_limits<u32>::max(),
                       direct_shared));
        const RangePlan exact = PlanRange(
            shape,
            Gpu(RangeSource::Metal, width_bits[width_index], width, 4u,
                exact_limit, std::numeric_limits<u32>::max(), direct_shared));
        const RangePlan above = PlanRange(
            shape, Gpu(RangeSource::Metal, width_bits[width_index], width, 4u,
                       exact_limit + 1u, std::numeric_limits<u32>::max(),
                       direct_shared));
        if (!below.ok() ||
            below.candidate().disposition() != RangePath::Direct ||
            !exact.ok() ||
            exact.candidate() !=
                *RangeCandidate::shared_halo(width, capacity) ||
            exact.cost().shared_bytes !=
                static_cast<u64>(width + 2u * capacity) * element_bytes ||
            !above.ok() || above.candidate() != exact.candidate()) {
          return false;
        }
      }
    }
  }
  return true;
}

[[nodiscard]] constexpr bool CandidateFamilyLegalityContract() {
  constexpr std::uint8_t direct_prefix =
      RangeSupportBit(RangeSupport::Direct) |
      RangeSupportBit(RangeSupport::PrefixDifference);
  constexpr std::uint8_t direct_block =
      RangeSupportBit(RangeSupport::Direct) |
      RangeSupportBit(RangeSupport::BlockPrefixSuffix);
  const RangeCaps prefix =
      Gpu(RangeSource::Metal, kRangeWidth64Bit, 64u, 4u, 32768u,
          std::numeric_limits<u32>::max(), direct_prefix);
  const RangeCaps block =
      Gpu(RangeSource::Vulkan, kRangeWidth64Bit, 64u, 0u, 0u,
          std::numeric_limits<u32>::max(), direct_block);
  const RangePlan sum = PlanRange(Shape(RangeOp::Sum, 4097u, 4097u), prefix);
  const RangePlan minimum_on_prefix =
      PlanRange(Shape(RangeOp::Minimum, 4097u, 4097u), prefix);
  const RangePlan maximum =
      PlanRange(Shape(RangeOp::Maximum, 4097u, 4097u), block);
  const RangePlan sum_on_block =
      PlanRange(Shape(RangeOp::Sum, 4097u, 4097u), block);
  const RangeTraits saturating =
      *RangeTraits::sum_saturating(ComputeDomain::Fixed);
  const RangePlan saturating_sum =
      PlanRange(*RangeShape::affine(saturating, RangeBoundary::Clamp, 4097u,
                                    4097u, 8195u, 1u, 4097u, 4u),
                prefix);
  return sum.ok() &&
         sum.candidate().disposition() == RangePath::PrefixDifference &&
         minimum_on_prefix.ok() &&
         minimum_on_prefix.candidate().disposition() == RangePath::Direct &&
         maximum.ok() &&
         maximum.candidate().disposition() == RangePath::BlockPrefixSuffix &&
         sum_on_block.ok() &&
         sum_on_block.candidate().disposition() == RangePath::Direct &&
         saturating_sum.ok() &&
         saturating_sum.candidate().disposition() == RangePath::Direct;
}

[[nodiscard]] constexpr bool ExhaustivePlannerContract() {
  struct DomainWidth final {
    ComputeDomain domain;
    u32 bytes;
  };
  constexpr std::array domain_widths{DomainWidth{ComputeDomain::I32, 4u},
                                     DomainWidth{ComputeDomain::U32, 4u},
                                     DomainWidth{ComputeDomain::Fixed, 4u},
                                     DomainWidth{ComputeDomain::I64, 8u},
                                     DomainWidth{ComputeDomain::U64, 8u},
                                     DomainWidth{ComputeDomain::Fixed, 8u}};
  constexpr std::array operations{RangeOp::Sum, RangeOp::Minimum,
                                  RangeOp::Maximum};
  constexpr std::array<u32, 3u> widths{64u, 128u, 256u};
  constexpr std::array<std::uint8_t, 3u> width_bits{
      kRangeWidth64Bit, kRangeWidth128Bit, kRangeWidth256Bit};

  for (std::size_t width_index = 0u; width_index < widths.size();
       ++width_index) {
    const u32 width = widths[width_index];
    const std::array<u64, 5u> counts{1u, width - 1u, width, width + 1u,
                                     2u * width + 13u};
    for (const DomainWidth domain_width : domain_widths) {
      const u64 shared_limit =
          static_cast<u64>(3u * width) * domain_width.bytes * 4u;
      for (const u64 count : counts) {
        const u64 groups = count / width + (count % width != 0u ? 1u : 0u);
        const RangeCaps capabilities =
            Gpu(RangeSource::Metal, width_bits[width_index], width, 4u,
                shared_limit, groups, kAllCandidates);
        const std::array<u64, 3u> radii{1u, count < 9u ? count : 9u, count};
        for (const u64 radius : radii) {
          for (const RangeOp operation : operations) {
            const RangePlan plan =
                PlanRange(Shape(operation, count, radius, domain_width.bytes,
                                domain_width.domain),
                          capabilities);
            const RangePath expected = radius <= width ? RangePath::SharedHalo
                                       : operation == RangeOp::Sum
                                           ? RangePath::PrefixDifference
                                           : RangePath::BlockPrefixSuffix;
            const std::uint8_t expected_legal = radius <= width ? 3u : 2u;
            if (!plan.ok() || plan.candidate().disposition() != expected ||
                plan.candidate().width() != width ||
                (expected == RangePath::SharedHalo &&
                 plan.candidate().radius_capacity() != width) ||
                plan.legal_candidate_count() != expected_legal ||
                plan.pareto_candidate_count() == 0u ||
                plan.pareto_candidate_count() > plan.legal_candidate_count() ||
                plan.stage(plan.stage_count() - 1u).groups > groups) {
              return false;
            }
          }
        }
      }
    }
  }
  return true;
}

[[nodiscard]] constexpr bool CapabilityBoundaryContract() {
  constexpr std::uint8_t direct_shared =
      RangeSupportBit(RangeSupport::Direct) |
      RangeSupportBit(RangeSupport::SharedHalo);
  constexpr u64 exact_limit = (64u + 2u * 9u) * 4u * 4u;
  const RangeShape shape = Shape(RangeOp::Sum, 65u, 9u);
  const RangePlan below =
      PlanRange(shape, Gpu(RangeSource::Vulkan, kRangeWidth64Bit, 64u, 4u,
                           exact_limit - 1u, std::numeric_limits<u32>::max(),
                           direct_shared));
  const RangePlan exact = PlanRange(
      shape, Gpu(RangeSource::Vulkan, kRangeWidth64Bit, 64u, 4u, exact_limit,
                 std::numeric_limits<u32>::max(), direct_shared));
  const RangeCaps direct_only =
      Gpu(RangeSource::Metal, kRangeWidth64Bit, 64u, 0u, 0u, 2u,
          RangeSupportBit(RangeSupport::Direct));
  const RangeCaps one_group =
      Gpu(RangeSource::Metal, kRangeWidth64Bit, 64u, 0u, 0u, 1u,
          RangeSupportBit(RangeSupport::Direct));
  const RangeCaps three_groups =
      Gpu(RangeSource::Metal, kRangeWidth64Bit, 64u, 0u, 0u, 3u,
          RangeSupportBit(RangeSupport::Direct));
  const RangePlan dispatch_below =
      PlanRange(Shape(RangeOp::Sum, 128u, 1u), one_group);
  const RangePlan dispatch_exact =
      PlanRange(Shape(RangeOp::Sum, 128u, 1u), direct_only);
  const RangePlan dispatch_above =
      PlanRange(Shape(RangeOp::Sum, 128u, 1u), three_groups);
  const RangePlan dispatch_over =
      PlanRange(Shape(RangeOp::Sum, 129u, 1u), direct_only);
  return below.ok() && below.candidate().disposition() == RangePath::Direct &&
         exact.ok() &&
         exact.candidate() == *RangeCandidate::shared_halo(64u, 9u) &&
         exact.cost().shared_bytes == (64u + 18u) * 4u &&
         !dispatch_below.ok() && dispatch_exact.ok() && dispatch_above.ok() &&
         !dispatch_over.ok() &&
         std::string_view{dispatch_over.reason()} ==
             "compute_range_aggregate_candidate_unavailable";
}

[[nodiscard]] constexpr bool CostCrossoverContract() {
  constexpr std::uint8_t direct_prefix =
      RangeSupportBit(RangeSupport::Direct) |
      RangeSupportBit(RangeSupport::PrefixDifference);
  constexpr std::uint8_t direct_block =
      RangeSupportBit(RangeSupport::Direct) |
      RangeSupportBit(RangeSupport::BlockPrefixSuffix);
  const RangeCaps prefix =
      Gpu(RangeSource::Metal, kRangeWidth64Bit, 64u, 4u, 32768u,
          std::numeric_limits<u32>::max(), direct_prefix);
  const RangeCaps block =
      Gpu(RangeSource::Vulkan, kRangeWidth64Bit, 64u, 0u, 0u,
          std::numeric_limits<u32>::max(), direct_block);
  const RangePlan sum_before = PlanRange(Shape(RangeOp::Sum, 515u, 1u), prefix);
  const RangePlan sum_after = PlanRange(Shape(RangeOp::Sum, 515u, 2u), prefix);
  const RangePlan minimum_before =
      PlanRange(Shape(RangeOp::Minimum, 515u, 1u), block);
  const RangePlan minimum_after =
      PlanRange(Shape(RangeOp::Minimum, 515u, 2u), block);
  return sum_before.ok() && sum_after.ok() && minimum_before.ok() &&
         minimum_after.ok() &&
         sum_before.candidate().disposition() == RangePath::Direct &&
         sum_after.candidate().disposition() == RangePath::PrefixDifference &&
         minimum_before.candidate().disposition() == RangePath::Direct &&
         minimum_after.candidate().disposition() ==
             RangePath::BlockPrefixSuffix;
}

[[nodiscard]] constexpr bool LinearWorkContract() {
  constexpr u64 count = 4097u;
  constexpr std::uint8_t direct_prefix =
      RangeSupportBit(RangeSupport::Direct) |
      RangeSupportBit(RangeSupport::PrefixDifference);
  constexpr std::uint8_t direct_block =
      RangeSupportBit(RangeSupport::Direct) |
      RangeSupportBit(RangeSupport::BlockPrefixSuffix);
  const RangeCaps prefix_capabilities =
      Gpu(RangeSource::Metal, kRangeWidth64Bit, 64u, 4u, 32768u,
          std::numeric_limits<u32>::max(), direct_prefix);
  const RangeCaps block_capabilities =
      Gpu(RangeSource::Vulkan, kRangeWidth64Bit, 64u, 0u, 0u,
          std::numeric_limits<u32>::max(), direct_block);
  const RangePlan prefix_half =
      PlanRange(Shape(RangeOp::Sum, count, count / 2u), prefix_capabilities);
  const RangePlan prefix_full =
      PlanRange(Shape(RangeOp::Sum, count, count), prefix_capabilities);
  const RangePlan block_half =
      PlanRange(Shape(RangeOp::Minimum, count, count / 2u), block_capabilities);
  const RangePlan block_full =
      PlanRange(Shape(RangeOp::Minimum, count, count), block_capabilities);
  return prefix_half.ok() && prefix_full.ok() && block_half.ok() &&
         block_full.ok() &&
         prefix_half.candidate().disposition() == RangePath::PrefixDifference &&
         prefix_full.candidate() == prefix_half.candidate() &&
         prefix_full.stage_count() == prefix_half.stage_count() &&
         prefix_full.temporary_count() == prefix_half.temporary_count() &&
         prefix_full.cost().scratch_bytes == prefix_half.cost().scratch_bytes &&
         prefix_full.cost().combine_ops <=
             static_cast<rund::kernel::u128>(8u) * count &&
         block_half.candidate().disposition() == RangePath::BlockPrefixSuffix &&
         block_full.candidate() == block_half.candidate() &&
         block_full.stage_count() == 2u &&
         block_full.cost().combine_ops <=
             static_cast<rund::kernel::u128>(7u) * count &&
         block_full.cost().scratch_bytes <= 6u * count * 4u;
}

[[nodiscard]] constexpr bool CpuCandidateContract() {
  const RangePlan reference_sum =
      PlanRange(Shape(RangeOp::Sum, 4097u, 4097u), RangeCaps::cpu_reference());
  const RangePlan prefix =
      PlanRange(Shape(RangeOp::Sum, 4097u, 4097u), RangeCaps::cpu());
  const RangePlan block =
      PlanRange(Shape(RangeOp::Maximum, 4097u, 4097u), RangeCaps::cpu());
  return reference_sum.ok() &&
         reference_sum.candidate() == RangeCandidate::direct_cpu() &&
         prefix.ok() &&
         prefix.candidate().disposition() == RangePath::PrefixDifference &&
         prefix.candidate().width() == kRangeCpuBlockWidth &&
         prefix.stage_count() == 2u && prefix.temporary_count() == 1u &&
         prefix.stage(0u).disposition == RangeStageKind::PrefixSequential &&
         prefix.stage(0u).width == 0u &&
         prefix.stage(1u).disposition == RangeStageKind::PrefixWindow &&
         prefix.temporary(0u).role == RangeTempRole::PrefixValues &&
         prefix.temporary(0u).bytes == 4097u * 4u &&
         prefix.cost().shared_bytes == 0u &&
         prefix.cost().launched_lanes == 8194u && block.ok() &&
         block.candidate().disposition() == RangePath::BlockPrefixSuffix &&
         block.candidate().width() == kRangeCpuBlockWidth &&
         block.stage_count() == 2u &&
         block.stage(0u) ==
             RangeStagePlan{.disposition = RangeStageKind::BlockPrefixSuffix,
                            .level = 0u,
                            .element_count = 12291u,
                            .groups = 1u,
                            .width = 0u} &&
         block.stage(1u) ==
             RangeStagePlan{.disposition = RangeStageKind::BlockWindow,
                            .level = 0u,
                            .element_count = 4097u,
                            .groups = 1u,
                            .width = 0u} &&
         block.cost().launched_lanes == 28679u;
}

static_assert(BasicSelectionContract());
static_assert(WidthAndCapacityContract());
static_assert(CandidateFamilyLegalityContract());
static_assert(CapabilityBoundaryContract());
static_assert(CostCrossoverContract());
static_assert(LinearWorkContract());
static_assert(CpuCandidateContract());

} // namespace

bool PlannerContract() {
  return BasicSelectionContract() && WidthAndCapacityContract() &&
         CandidateFamilyLegalityContract() && ExhaustivePlannerContract() &&
         CapabilityBoundaryContract() && CostCrossoverContract() &&
         LinearWorkContract() && CpuCandidateContract();
}

} // namespace node_accel_contract::range

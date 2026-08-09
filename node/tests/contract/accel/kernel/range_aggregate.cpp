#include "src/accel/metal/range/local.hpp"
#include "src/accel/range_aggregate/execution.hpp"
#include "src/accel/range_aggregate/plan.hpp"
#include "src/accel/source/hash.hpp"
#include "src/accel/vulkan/range/local.hpp"

#include <array>
#include <cstdint>
#include <limits>
#include <optional>
#include <string>
#include <string_view>

namespace {

using rund::kernel::ComputeDomain;
using rund::kernel::u32;
using rund::kernel::u64;
using namespace rund::node::accel::detail;

inline constexpr std::uint8_t kAllRangeCandidates =
    RangeSupportBit(RangeSupport::Direct) |
    RangeSupportBit(RangeSupport::SharedHalo) |
    RangeSupportBit(RangeSupport::PrefixDifference) |
    RangeSupportBit(RangeSupport::BlockPrefixSuffix);

[[nodiscard]] constexpr RangeTraits
Traits(const RangeOp operation,
       const ComputeDomain domain = ComputeDomain::U32) {
  const std::optional<RangeTraits> traits = RangeTraits::make(
      operation, domain,
      operation == RangeOp::Sum ? RangeLaw::ModuloWidth : RangeLaw::OrderOnly);
  return *traits;
}

[[nodiscard]] constexpr RangeShape
Shape(const RangeOp operation, const u64 count, const u64 radius,
      const u32 element_bytes = 4u,
      const ComputeDomain domain = ComputeDomain::U32) {
  return *RangeShape::window(Traits(operation, domain), RangeBoundary::Clamp,
                             count, radius, element_bytes);
}

[[nodiscard]] constexpr RangeShape
AffineShape(const RangeOp operation, const RangeBoundary boundary,
            const u64 input_count, const u64 output_count,
            const u64 window_size, const u64 stride, const u64 padding,
            const u32 element_bytes = 4u,
            const ComputeDomain domain = ComputeDomain::U32) {
  return *RangeShape::affine(Traits(operation, domain), boundary, input_count,
                             output_count, window_size, stride, padding,
                             element_bytes);
}

[[nodiscard]] constexpr RangeCaps
Gpu(const RangeSource variant, const std::uint8_t widths = kRangeKnownWidthMask,
    const u32 maximum_threads = 256u, const u32 occupancy = 4u,
    const u64 shared_limit = 32768u,
    const u64 maximum_groups = std::numeric_limits<u32>::max(),
    const std::uint8_t support = kAllRangeCandidates,
    const u64 maximum_storage_elements = 0u,
    const u64 maximum_storage_bytes = std::numeric_limits<u64>::max()) {
  const u64 storage_limit =
      maximum_storage_elements != 0u
          ? maximum_storage_elements
          : (variant == RangeSource::Vulkan ? std::numeric_limits<u32>::max()
                                            : std::numeric_limits<u64>::max());
  return *RangeCaps::gpu(variant, widths, maximum_threads, occupancy,
                         shared_limit, maximum_groups, storage_limit,
                         maximum_storage_bytes, support);
}

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

[[nodiscard]] constexpr bool AlgebraContract() {
  constexpr std::array domains{ComputeDomain::I32, ComputeDomain::U32,
                               ComputeDomain::I64, ComputeDomain::U64,
                               ComputeDomain::Fixed};
  for (const ComputeDomain domain : domains) {
    const RangeTraits sum = Traits(RangeOp::Sum, domain);
    const RangeTraits minimum = Traits(RangeOp::Minimum, domain);
    const RangeTraits maximum = Traits(RangeOp::Maximum, domain);
    const std::optional<RangeTraits> saturating =
        RangeTraits::sum_saturating(domain);
    if (!sum.associative() || !sum.has_identity() || !sum.commutative() ||
        !sum.invertible() || sum.idempotent() || sum.ordered() ||
        !minimum.associative() || !minimum.has_identity() ||
        !minimum.commutative() || minimum.invertible() ||
        !minimum.idempotent() || !minimum.ordered() || maximum.invertible() ||
        !maximum.idempotent() || !maximum.ordered()) {
      return false;
    }
    const bool signed_domain = domain == ComputeDomain::I32 ||
                               domain == ComputeDomain::I64 ||
                               domain == ComputeDomain::Fixed;
    const bool unsigned_domain =
        domain == ComputeDomain::U32 || domain == ComputeDomain::U64;
    if (saturating.has_value() != signed_domain ||
        (saturating.has_value() &&
         (saturating->associative() || saturating->invertible() ||
          !saturating->has_identity()))) {
      return false;
    }
    if (sum.signed_domain() != signed_domain ||
        sum.unsigned_domain() != unsigned_domain ||
        sum.fixed_domain() != (domain == ComputeDomain::Fixed)) {
      return false;
    }
  }

  return true;
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
  const RangePlan saturating_sum = PlanRange(
      *RangeShape::window(saturating, RangeBoundary::Clamp, 4097u, 4097u, 4u),
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
                shared_limit, groups, kAllRangeCandidates);
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

[[nodiscard]] constexpr bool StorageIndexCapabilityContract() {
  constexpr u64 exact = std::numeric_limits<u32>::max();
  constexpr std::uint8_t direct = RangeSupportBit(RangeSupport::Direct);
  constexpr std::uint8_t direct_block =
      direct | RangeSupportBit(RangeSupport::BlockPrefixSuffix);
  const RangeCaps vulkan =
      Gpu(RangeSource::Vulkan, kRangeWidth64Bit, 64u, 0u, 0u,
          std::numeric_limits<u32>::max(), direct, exact);
  const std::optional<RangeCaps> oversized_vulkan_capability =
      RangeCaps::gpu(RangeSource::Vulkan, kRangeWidth64Bit, 64u, 0u, 0u,
                     std::numeric_limits<u32>::max(), exact + 1u, direct);
  const RangePlan below =
      PlanRange(Shape(RangeOp::Sum, exact - 1u, 1u), vulkan);
  const RangePlan at = PlanRange(Shape(RangeOp::Sum, exact, 1u), vulkan);
  const RangePlan above =
      PlanRange(Shape(RangeOp::Sum, exact + 1u, 1u), vulkan);
  const RangePlan input_only_above =
      PlanRange(AffineShape(RangeOp::Sum, RangeBoundary::Clamp, exact + 1u, 1u,
                            1u, 1u, 0u),
                vulkan);
  const RangePlan metal_above =
      PlanRange(Shape(RangeOp::Sum, exact + 1u, 1u),
                Gpu(RangeSource::Metal, kRangeWidth64Bit, 64u, 0u, 0u,
                    std::numeric_limits<u32>::max(), direct,
                    std::numeric_limits<u64>::max()));
  const RangePlan cpu_above = PlanRange(Shape(RangeOp::Sum, exact + 1u, 1u),
                                        RangeCaps::cpu_reference());
  const RangePlan identity_wide =
      PlanRange(Shape(RangeOp::Sum, 65u, 1u), vulkan);
  const RangePlan identity_narrow =
      PlanRange(Shape(RangeOp::Sum, 65u, 1u),
                Gpu(RangeSource::Vulkan, kRangeWidth64Bit, 64u, 0u, 0u,
                    std::numeric_limits<u32>::max(), direct, exact - 1u));

  const RangeCaps vulkan_block =
      Gpu(RangeSource::Vulkan, kRangeWidth64Bit, 64u, 0u, 0u,
          std::numeric_limits<u32>::max(), direct_block, exact);
  const RangePlan span_at = PlanRange(
      AffineShape(RangeOp::Minimum, RangeBoundary::Clip, 1u, 1u, exact, 1u, 0u),
      vulkan_block);
  const RangePlan span_above =
      PlanRange(AffineShape(RangeOp::Minimum, RangeBoundary::Clip, 1u, 1u,
                            exact + 1u, 1u, 0u),
                vulkan_block);
  const std::optional<RangeExec> at_execution = RangeExec::from(at);
  return !oversized_vulkan_capability.has_value() &&
         vulkan.maximum_storage_element_count() == exact && below.ok() &&
         at.ok() && at_execution.has_value() &&
         at_execution->vulkan_dispatch_fits(std::numeric_limits<u32>::max()) &&
         !above.ok() &&
         std::string_view{above.reason()} ==
             "compute_range_aggregate_candidate_unavailable" &&
         !input_only_above.ok() && metal_above.ok() && cpu_above.ok() &&
         identity_wide.ok() && identity_narrow.ok() &&
         identity_wide.source_identity() == identity_narrow.source_identity() &&
         identity_wide.execution_identity() ==
             identity_narrow.execution_identity() &&
         span_at.ok() && span_at.legal_candidate_count() == 2u &&
         span_above.ok() && span_above.legal_candidate_count() == 1u &&
         span_above.candidate().disposition() == RangePath::Direct;
}

[[nodiscard]] constexpr bool StorageBindingCapabilityContract() {
  constexpr std::uint8_t direct_prefix =
      RangeSupportBit(RangeSupport::Direct) |
      RangeSupportBit(RangeSupport::PrefixDifference);
  constexpr std::uint8_t direct_block =
      RangeSupportBit(RangeSupport::Direct) |
      RangeSupportBit(RangeSupport::BlockPrefixSuffix);

  const RangeShape prefix_shape =
      AffineShape(RangeOp::Sum, RangeBoundary::Clip, 65u, 65u, 65u, 1u, 0u);
  const RangePlan prefix_below =
      PlanRange(prefix_shape,
                Gpu(RangeSource::Metal, kRangeWidth64Bit, 64u, 4u, 32768u,
                    std::numeric_limits<u32>::max(), direct_prefix, 0u, 259u));
  const RangePlan prefix_exact =
      PlanRange(prefix_shape,
                Gpu(RangeSource::Metal, kRangeWidth64Bit, 64u, 4u, 32768u,
                    std::numeric_limits<u32>::max(), direct_prefix, 0u, 260u));
  const RangePlan prefix_above =
      PlanRange(prefix_shape,
                Gpu(RangeSource::Metal, kRangeWidth64Bit, 64u, 4u, 32768u,
                    std::numeric_limits<u32>::max(), direct_prefix, 0u, 261u));

  const RangeShape block_shape = AffineShape(
      RangeOp::Minimum, RangeBoundary::Clip, 16u, 16u, 33u, 1u, 16u);
  const RangePlan block_below =
      PlanRange(block_shape,
                Gpu(RangeSource::Vulkan, kRangeWidth64Bit, 64u, 0u, 0u,
                    std::numeric_limits<u32>::max(), direct_block, 0u, 191u));
  const RangePlan block_exact =
      PlanRange(block_shape,
                Gpu(RangeSource::Vulkan, kRangeWidth64Bit, 64u, 0u, 0u,
                    std::numeric_limits<u32>::max(), direct_block, 0u, 192u));
  const RangePlan block_above =
      PlanRange(block_shape,
                Gpu(RangeSource::Vulkan, kRangeWidth64Bit, 64u, 0u, 0u,
                    std::numeric_limits<u32>::max(), direct_block, 0u, 193u));

  return prefix_below.ok() &&
         prefix_below.candidate().disposition() == RangePath::Direct &&
         prefix_exact.ok() &&
         prefix_exact.candidate().disposition() ==
             RangePath::PrefixDifference &&
         prefix_above.ok() &&
         prefix_above.candidate() == prefix_exact.candidate() &&
         prefix_above.source_identity() == prefix_exact.source_identity() &&
         prefix_above.execution_identity() ==
             prefix_exact.execution_identity() &&
         block_below.ok() &&
         block_below.candidate().disposition() == RangePath::Direct &&
         block_exact.ok() &&
         block_exact.candidate().disposition() ==
             RangePath::BlockPrefixSuffix &&
         block_above.ok() &&
         block_above.candidate() == block_exact.candidate() &&
         block_exact.temporary(0u).bytes == 192u &&
         block_exact.temporary(1u).bytes == 192u;
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

[[nodiscard]] constexpr bool PrefixHierarchyContract() {
  constexpr std::uint8_t direct_prefix =
      RangeSupportBit(RangeSupport::Direct) |
      RangeSupportBit(RangeSupport::PrefixDifference);
  const RangeCaps capabilities =
      Gpu(RangeSource::Metal, kRangeWidth64Bit, 64u, 4u, 32768u,
          std::numeric_limits<u32>::max(), direct_prefix);
  const RangePlan plan =
      PlanRange(Shape(RangeOp::Sum, 4097u, 4097u), capabilities);
  constexpr std::array expected{
      RangeStageKind::PrefixBlock,   RangeStageKind::PrefixSummary,
      RangeStageKind::PrefixSummary, RangeStageKind::PrefixFixup,
      RangeStageKind::PrefixFixup,   RangeStageKind::PrefixWindow,
  };
  if (!plan.ok() ||
      plan.candidate().disposition() != RangePath::PrefixDifference ||
      plan.stage_count() != expected.size() || plan.temporary_count() != 3u ||
      plan.cost().scratch_bytes != 16656u ||
      plan.cost().dispatch_count != expected.size()) {
    return false;
  }
  for (std::size_t index = 0u; index < expected.size(); ++index) {
    if (plan.stage(index).disposition != expected[index]) {
      return false;
    }
  }
  return plan.temporary(0u).role == RangeTempRole::PrefixValues &&
         plan.temporary(0u).first_stage == 0u &&
         plan.temporary(0u).last_stage == 5u &&
         plan.temporary(1u).role == RangeTempRole::BlockSummaries &&
         plan.temporary(1u).bytes == 65u * 4u &&
         plan.temporary(1u).last_stage == 4u &&
         plan.temporary(2u).bytes == 2u * 4u &&
         plan.temporary(2u).last_stage == 3u;
}

[[nodiscard]] constexpr bool PrefixStageSubstrateContract() {
  constexpr auto hierarchy =
      PlanRangePrefixTree(4097u, 64u, 4u, std::numeric_limits<u32>::max());
  constexpr auto flat_single = PlanRangeFlatPrefix(256u, 1u, 128u, 4u);
  constexpr auto flat_multiple = PlanRangeFlatPrefix(513u, 3u, 128u, 8u);
  constexpr auto invalid = PlanRangeFlatPrefix(0u, 0u, 128u, 4u);
  constexpr auto dispatch_limited = PlanRangePrefixTree(4097u, 64u, 4u, 64u);
  return hierarchy.ok() &&
         hierarchy.disposition() == RangePrefixKind::Hierarchical &&
         hierarchy.stage_count() == 5u && hierarchy.temporary_count() == 2u &&
         hierarchy.stage(0u) ==
             RangeStagePlan{.disposition = RangeStageKind::PrefixBlock,
                            .level = 0u,
                            .element_count = 4097u,
                            .groups = 65u,
                            .width = 64u} &&
         hierarchy.stage(4u) ==
             RangeStagePlan{.disposition = RangeStageKind::PrefixFixup,
                            .level = 0u,
                            .element_count = 4097u,
                            .groups = 65u,
                            .width = 64u} &&
         hierarchy.temporary(0u).bytes == 260u &&
         hierarchy.temporary(0u).last_stage == 4u &&
         hierarchy.temporary(1u).bytes == 8u &&
         hierarchy.temporary(1u).last_stage == 3u && flat_single.ok() &&
         flat_single.disposition() == RangePrefixKind::FlatBlockTotals &&
         flat_single.stage_count() == 1u &&
         flat_single.temporary_count() == 1u &&
         flat_single.stage(0u).groups == 1u &&
         flat_single.temporary(0u).bytes == 4u &&
         flat_single.temporary(0u).last_stage == 0u && flat_multiple.ok() &&
         flat_multiple.disposition() == RangePrefixKind::FlatBlockTotals &&
         flat_multiple.stage_count() == 3u &&
         flat_multiple.stage(0u).groups == 3u &&
         flat_multiple.stage(1u).groups == 1u &&
         flat_multiple.stage(2u).groups == 3u &&
         flat_multiple.temporary(0u).bytes == 24u &&
         flat_multiple.temporary(0u).last_stage == 2u && !invalid.ok() &&
         !dispatch_limited.ok();
}

[[nodiscard]] constexpr bool BlockPrefixSuffixContract() {
  constexpr std::uint8_t direct_block =
      RangeSupportBit(RangeSupport::Direct) |
      RangeSupportBit(RangeSupport::BlockPrefixSuffix);
  const RangeCaps capabilities =
      Gpu(RangeSource::Vulkan, kRangeWidth64Bit, 64u, 0u, 0u,
          std::numeric_limits<u32>::max(), direct_block);
  const RangePlan plan =
      PlanRange(Shape(RangeOp::Minimum, 4097u, 4097u), capabilities);
  return plan.ok() &&
         plan.candidate().disposition() == RangePath::BlockPrefixSuffix &&
         plan.stage_count() == 2u && plan.temporary_count() == 2u &&
         plan.stage(0u) ==
             RangeStagePlan{.disposition = RangeStageKind::BlockPrefixSuffix,
                            .level = 0u,
                            .element_count = 12291u,
                            .groups = 1u,
                            .width = 64u} &&
         plan.stage(1u).disposition == RangeStageKind::BlockWindow &&
         plan.stage(1u).groups == 65u && plan.cost().scratch_bytes == 98328u &&
         plan.cost().dispatch_count == 2u;
}

[[nodiscard]] constexpr bool IdentityContract() {
  constexpr std::uint8_t direct_prefix =
      RangeSupportBit(RangeSupport::Direct) |
      RangeSupportBit(RangeSupport::PrefixDifference);
  const RangeCaps metal =
      Gpu(RangeSource::Metal, kRangeWidth256Bit, 256u, 4u, 32768u,
          std::numeric_limits<u32>::max(), direct_prefix);
  const RangeCaps vulkan =
      Gpu(RangeSource::Vulkan, kRangeWidth256Bit, 256u, 4u, 32768u,
          std::numeric_limits<u32>::max(), direct_prefix);
  const RangePlan first = PlanRange(Shape(RangeOp::Sum, 515u, 300u), metal);
  const RangePlan second = PlanRange(Shape(RangeOp::Sum, 515u, 515u), metal);
  const RangePlan other_backend =
      PlanRange(Shape(RangeOp::Sum, 515u, 300u), vulkan);
  return first.ok() && second.ok() && other_backend.ok() &&
         first.candidate() == second.candidate() &&
         first.source_identity() == second.source_identity() &&
         first.execution_identity() != second.execution_identity() &&
         first.source_identity() != other_backend.source_identity();
}

[[nodiscard]] bool ExecutionProjectionContract() {
  constexpr std::uint8_t direct_shared =
      RangeSupportBit(RangeSupport::Direct) |
      RangeSupportBit(RangeSupport::SharedHalo);
  constexpr std::uint8_t direct_prefix =
      RangeSupportBit(RangeSupport::Direct) |
      RangeSupportBit(RangeSupport::PrefixDifference);
  constexpr std::uint8_t direct_block =
      RangeSupportBit(RangeSupport::Direct) |
      RangeSupportBit(RangeSupport::BlockPrefixSuffix);
  const RangeCaps shared_caps =
      Gpu(RangeSource::Metal, kRangeWidth128Bit, 128u, 4u,
          (128u + 2u * 9u) * 4u * 4u, std::numeric_limits<u32>::max(),
          direct_shared);
  const RangeCaps prefix_caps =
      Gpu(RangeSource::Vulkan, kRangeWidth64Bit, 64u, 4u, 32768u,
          std::numeric_limits<u32>::max(), direct_prefix);
  const RangeCaps block_caps =
      Gpu(RangeSource::Metal, kRangeWidth64Bit, 64u, 0u, 0u,
          std::numeric_limits<u32>::max(), direct_block);
  const RangePlan shared =
      PlanRange(Shape(RangeOp::Sum, 129u, 9u), shared_caps);
  const RangePlan prefix =
      PlanRange(Shape(RangeOp::Sum, 4097u, 4097u), prefix_caps);
  const RangePlan block =
      PlanRange(Shape(RangeOp::Maximum, 4097u, 4097u), block_caps);
  const RangePlan cpu =
      PlanRange(Shape(RangeOp::Sum, 17u, 3u), RangeCaps::cpu_reference());
  const auto shared_exec = RangeExec::from(shared);
  const auto prefix_exec = RangeExec::from(prefix);
  const auto block_exec = RangeExec::from(block);
  if (!shared_exec.has_value() || !prefix_exec.has_value() ||
      !block_exec.has_value() || RangeExec::from(cpu).has_value()) {
    return false;
  }
  const auto shared_params = shared_exec->stage_params(0u);
  const auto prefix_scratch = prefix_exec->stage_scratch(0u);
  const auto block_scratch = block_exec->stage_scratch(0u);
  return shared_exec->shape() == *RangeGpuShape::shared_halo(128u, 9u) &&
         prefix_exec->shape() == *RangeGpuShape::direct(64u) &&
         block_exec->shape() == *RangeGpuShape::direct(64u) &&
         shared_exec->descriptor_count() == 3u &&
         prefix_exec->descriptor_count() == 5u &&
         block_exec->descriptor_count() == 5u &&
         shared_exec->static_shared_bytes() == (128u + 18u) * 4u &&
         prefix_exec->static_shared_bytes() == 64u * 4u &&
         block_exec->static_shared_bytes() == 0u && shared_params.has_value() &&
         shared_params->stage() ==
             static_cast<u32>(RangeStageKind::SharedHalo) &&
         !shared_exec->stage_scratch(0u)->uses_global_scratch() &&
         prefix_scratch.has_value() && prefix_scratch->uses_global_scratch() &&
         prefix_scratch->first().role() == RangeTempRole::PrefixValues &&
         block_scratch.has_value() && block_scratch->uses_global_scratch() &&
         block_scratch->first().role() == RangeTempRole::ForwardValues &&
         block_scratch->second().role() == RangeTempRole::BackwardValues &&
         shared_exec->source_identity() == shared.source_identity() &&
         prefix_exec->execution_identity() == prefix.execution_identity();
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

[[nodiscard]] constexpr bool AffineShapeContract() {
  const RangeShape centered = Shape(RangeOp::Sum, 17u, 7u);
  const RangeShape projected =
      AffineShape(RangeOp::Sum, RangeBoundary::Clamp, 17u, 17u, 15u, 1u, 7u);
  const RangeShape strided =
      AffineShape(RangeOp::Minimum, RangeBoundary::Clip, 19u, 7u, 9u, 3u, 2u);
  const RangeShape padded_anchor =
      AffineShape(RangeOp::Maximum, RangeBoundary::Clamp, 5u, 2u, 10u, 8u, 5u);
  const RangeShape wide_window =
      AffineShape(RangeOp::Minimum, RangeBoundary::Clip, 2u, 1u, 6u, 1u, 1u);
  const auto no_intersection = RangeShape::affine(
      Traits(RangeOp::Sum), RangeBoundary::Clip, 5u, 2u, 2u, 10u, 0u, 4u);
  const auto invalid_padding = RangeShape::affine(
      Traits(RangeOp::Sum), RangeBoundary::Clamp, 5u, 1u, 3u, 1u, 3u, 4u);
  const auto overflowing_anchor =
      RangeShape::affine(Traits(RangeOp::Sum), RangeBoundary::Clamp, 2u, 3u, 2u,
                         std::numeric_limits<u64>::max(), 1u, 4u);
  constexpr u64 span_input = std::numeric_limits<u64>::max() / 4u;
  constexpr u64 span_window = 2u * span_input + 1u;
  constexpr u64 span_padding = span_window - 1u;
  const auto span_overflow = RangeShape::affine(
      Traits(RangeOp::Minimum), RangeBoundary::Clamp, span_input, 2u,
      span_window, span_input + span_padding - 1u, span_padding, 4u);
  return centered.input_count() == projected.input_count() &&
         centered.output_count() == projected.output_count() &&
         centered.window_size() == projected.window_size() &&
         centered.stride() == projected.stride() &&
         centered.padding() == projected.padding() &&
         centered.centered_clamp() && projected.centered_clamp() &&
         !strided.centered_clamp() && strided.input_count() == 19u &&
         strided.output_count() == 7u && strided.window_size() == 9u &&
         strided.stride() == 3u && strided.padding() == 2u &&
         strided.right_extent() == 6u && *strided.affine_span() == 27u &&
         padded_anchor.valid() && wide_window.valid() &&
         wide_window.window_size() > 2u * wide_window.input_count() + 1u &&
         !no_intersection.has_value() && !invalid_padding.has_value() &&
         !overflowing_anchor.has_value() && span_overflow.has_value() &&
         !span_overflow->affine_span().has_value();
}

[[nodiscard]] constexpr bool AffineCostAndTopologyContract() {
  constexpr std::uint8_t direct = RangeSupportBit(RangeSupport::Direct);
  constexpr std::uint8_t direct_shared =
      direct | RangeSupportBit(RangeSupport::SharedHalo);
  constexpr std::uint8_t direct_prefix =
      direct | RangeSupportBit(RangeSupport::PrefixDifference);
  constexpr std::uint8_t direct_block =
      direct | RangeSupportBit(RangeSupport::BlockPrefixSuffix);
  const RangeCaps direct_caps =
      Gpu(RangeSource::Metal, kRangeWidth64Bit, 64u, 0u, 0u,
          std::numeric_limits<u32>::max(), direct);
  const RangeShape clamp =
      AffineShape(RangeOp::Sum, RangeBoundary::Clamp, 5u, 3u, 3u, 2u, 1u);
  const RangeShape clip =
      AffineShape(RangeOp::Sum, RangeBoundary::Clip, 5u, 3u, 3u, 2u, 1u);
  const RangePlan clamp_direct = PlanRange(clamp, direct_caps);
  const RangePlan clip_direct = PlanRange(clip, direct_caps);
  const RangeShape prefix_shape =
      AffineShape(RangeOp::Sum, RangeBoundary::Clip, 257u, 65u, 129u, 2u, 64u);
  const RangePlan prefix = PlanRange(
      prefix_shape, Gpu(RangeSource::Metal, kRangeWidth64Bit, 64u, 4u, 32768u,
                        std::numeric_limits<u32>::max(), direct_prefix));
  const RangeShape block_shape = AffineShape(
      RangeOp::Minimum, RangeBoundary::Clip, 101u, 7u, 50u, 3u, 10u);
  const RangeCaps block_caps =
      Gpu(RangeSource::Vulkan, kRangeWidth64Bit, 64u, 0u, 0u,
          std::numeric_limits<u32>::max(), direct_block);
  const RangePlan block = PlanRange(block_shape, block_caps);
  const RangeShape block_clamp_shape = AffineShape(
      RangeOp::Minimum, RangeBoundary::Clamp, 257u, 65u, 129u, 2u, 64u);
  const RangeShape block_clip_shape = AffineShape(
      RangeOp::Minimum, RangeBoundary::Clip, 257u, 65u, 129u, 2u, 64u);
  const RangePlan block_clamp = PlanRange(block_clamp_shape, block_caps);
  const RangePlan block_clip = PlanRange(block_clip_shape, block_caps);
  const RangePlan shared_for_affine = PlanRange(
      AffineShape(RangeOp::Sum, RangeBoundary::Clamp, 65u, 32u, 3u, 2u, 1u),
      Gpu(RangeSource::Metal, kRangeWidth64Bit, 64u, 4u, 32768u,
          std::numeric_limits<u32>::max(), direct_shared));
  return clamp_direct.ok() && clip_direct.ok() &&
         clamp_direct.cost().global_read_bytes == 9u * 4u &&
         clamp_direct.cost().global_write_bytes == 3u * 4u &&
         clamp_direct.cost().combine_ops == 6u &&
         clip_direct.cost().global_read_bytes == 7u * 4u &&
         clip_direct.cost().global_write_bytes == 3u * 4u &&
         clip_direct.cost().combine_ops == 4u && prefix.ok() &&
         prefix.candidate().disposition() == RangePath::PrefixDifference &&
         prefix.stage(prefix.stage_count() - 1u).element_count == 65u &&
         prefix.stage(prefix.stage_count() - 1u).groups == 2u &&
         prefix.temporary(0u).bytes == 257u * 4u && block.ok() &&
         block.candidate().disposition() == RangePath::BlockPrefixSuffix &&
         *block_shape.affine_span() == 68u &&
         block.stage(0u).element_count == 68u &&
         block.stage(1u).element_count == 7u &&
         block.cost().scratch_bytes == 2u * 68u * 4u && block_clamp.ok() &&
         block_clip.ok() &&
         block_clamp.candidate().disposition() ==
             RangePath::BlockPrefixSuffix &&
         block_clip.candidate() == block_clamp.candidate() &&
         block_clamp.cost().global_read_bytes == (2u * 257u + 2u * 65u) * 4u &&
         block_clip.cost().global_read_bytes ==
             (2u * (257u - 64u) + 2u * 65u) * 4u &&
         shared_for_affine.ok() &&
         shared_for_affine.candidate().disposition() != RangePath::SharedHalo;
}

[[nodiscard]] constexpr bool AffinePaddedDirectContract() {
  constexpr std::uint8_t direct = RangeSupportBit(RangeSupport::Direct);
  const RangeShape clamp =
      AffineShape(RangeOp::Sum, RangeBoundary::Clamp, 3u, 2u, 5u, 5u, 4u);
  const RangeShape clip =
      AffineShape(RangeOp::Sum, RangeBoundary::Clip, 3u, 2u, 5u, 5u, 4u);
  const RangePlan clamp_plan =
      PlanRange(clamp, Gpu(RangeSource::Metal, kRangeWidth64Bit, 64u, 0u, 0u,
                           std::numeric_limits<u32>::max(), direct));
  const RangePlan clip_plan =
      PlanRange(clip, Gpu(RangeSource::Vulkan, kRangeWidth64Bit, 64u, 0u, 0u,
                          std::numeric_limits<u32>::max(), direct));
  constexpr std::array<u32, 3u> input{2u, 5u, 7u};
  constexpr std::array<u32, 2u> clamp_expected{10u, 33u};
  constexpr std::array<u32, 2u> clip_expected{2u, 12u};
  std::array<u32, 2u> clamp_actual{};
  std::array<u32, 2u> clip_actual{};
  for (u64 output = 0u; output < 2u; ++output) {
    const u64 anchor = output * 5u;
    for (u64 slot = 0u; slot < 5u; ++slot) {
      u64 index = 0u;
      bool valid = true;
      if (slot < 4u) {
        const u64 delta = 4u - slot;
        if (anchor < delta) {
          index = 0u;
        } else {
          index = anchor - delta;
          if (index >= input.size()) {
            index = input.size() - 1u;
          }
        }
      } else if (anchor >= input.size() || slot - 4u >= input.size() - anchor) {
        index = input.size() - 1u;
      } else {
        index = anchor + slot - 4u;
      }
      clamp_actual[output] += input[index];

      if (slot < 4u) {
        const u64 delta = 4u - slot;
        if (anchor < delta) {
          valid = false;
        } else {
          index = anchor - delta;
          valid = index < input.size();
        }
      } else if (anchor >= input.size() || slot - 4u >= input.size() - anchor) {
        valid = false;
      } else {
        index = anchor + slot - 4u;
      }
      if (valid) {
        clip_actual[output] += input[index];
      }
    }
  }
  return clamp_plan.ok() && clip_plan.ok() &&
         clamp_plan.candidate().disposition() == RangePath::Direct &&
         clip_plan.candidate().disposition() == RangePath::Direct &&
         clamp_plan.cost().global_read_bytes == 10u * 4u &&
         clip_plan.cost().global_read_bytes == 3u * 4u &&
         clamp_actual == clamp_expected && clip_actual == clip_expected;
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

[[nodiscard]] constexpr bool AffineIdentityContract() {
  constexpr std::uint8_t direct_prefix =
      RangeSupportBit(RangeSupport::Direct) |
      RangeSupportBit(RangeSupport::PrefixDifference);
  const RangeCaps caps =
      Gpu(RangeSource::Metal, kRangeWidth64Bit, 64u, 4u, 32768u,
          std::numeric_limits<u32>::max(), direct_prefix);
  const RangePlan first = PlanRange(
      AffineShape(RangeOp::Sum, RangeBoundary::Clamp, 257u, 65u, 129u, 2u, 64u),
      caps);
  const RangePlan second =
      PlanRange(AffineShape(RangeOp::Sum, RangeBoundary::Clamp, 513u, 129u,
                            257u, 2u, 128u),
                caps);
  const RangePlan clipped = PlanRange(
      AffineShape(RangeOp::Sum, RangeBoundary::Clip, 257u, 65u, 129u, 2u, 64u),
      caps);
  return first.ok() && second.ok() && clipped.ok() &&
         first.candidate() == second.candidate() &&
         first.source_identity() == second.source_identity() &&
         first.execution_identity() != second.execution_identity() &&
         first.source_identity() != clipped.source_identity();
}

[[nodiscard]] bool AffineSourceContract() {
  constexpr std::uint8_t direct = RangeSupportBit(RangeSupport::Direct);
  constexpr std::uint8_t direct_prefix =
      direct | RangeSupportBit(RangeSupport::PrefixDifference);
  constexpr std::uint8_t direct_block =
      direct | RangeSupportBit(RangeSupport::BlockPrefixSuffix);
  const RangePlan direct_clip = PlanRange(
      AffineShape(RangeOp::Sum, RangeBoundary::Clip, 3u, 2u, 5u, 5u, 4u),
      Gpu(RangeSource::Metal, kRangeWidth64Bit, 64u, 0u, 0u,
          std::numeric_limits<u32>::max(), direct));
  const RangePlan direct_clamp = PlanRange(
      AffineShape(RangeOp::Sum, RangeBoundary::Clamp, 3u, 2u, 5u, 5u, 4u),
      Gpu(RangeSource::Metal, kRangeWidth64Bit, 64u, 0u, 0u,
          std::numeric_limits<u32>::max(), direct));
  const RangePlan prefix_clip = PlanRange(
      AffineShape(RangeOp::Sum, RangeBoundary::Clip, 257u, 65u, 129u, 2u, 64u),
      Gpu(RangeSource::Metal, kRangeWidth64Bit, 64u, 4u, 32768u,
          std::numeric_limits<u32>::max(), direct_prefix));
  const RangePlan block_clip =
      PlanRange(AffineShape(RangeOp::Minimum, RangeBoundary::Clip, 101u, 7u,
                            50u, 3u, 10u, 4u, ComputeDomain::I32),
                Gpu(RangeSource::Metal, kRangeWidth64Bit, 64u, 0u, 0u,
                    std::numeric_limits<u32>::max(), direct_block));
  const std::optional<RangeTraits> saturating =
      RangeTraits::sum_saturating(ComputeDomain::Fixed);
  const std::optional<RangeShape> saturating_shape =
      saturating.has_value()
          ? RangeShape::affine(*saturating, RangeBoundary::Clamp, 5u, 3u, 3u,
                               2u, 1u, 4u)
          : std::nullopt;
  const RangePlan saturating_direct =
      saturating_shape.has_value()
          ? PlanRange(*saturating_shape,
                      Gpu(RangeSource::Metal, kRangeWidth64Bit, 64u, 0u, 0u,
                          std::numeric_limits<u32>::max(), direct))
          : RangePlan::rejected("compute_range_aggregate_shape_invalid");
  const auto direct_exec = RangeExec::from(direct_clip);
  const auto direct_clamp_exec = RangeExec::from(direct_clamp);
  const auto prefix_exec = RangeExec::from(prefix_clip);
  const auto block_exec = RangeExec::from(block_clip);
  const auto saturating_exec = RangeExec::from(saturating_direct);
  if (!direct_exec.has_value() || !direct_clamp_exec.has_value() ||
      !prefix_exec.has_value() || !block_exec.has_value() ||
      !saturating_exec.has_value()) {
    return false;
  }
  const std::string direct_source = MetalRangeSource(*direct_exec);
  const std::string direct_clamp_source = MetalRangeSource(*direct_clamp_exec);
  const std::string prefix_source = MetalRangeSource(*prefix_exec);
  const std::string block_source = MetalRangeSource(*block_exec);
  const std::string saturating_source = MetalRangeSource(*saturating_exec);
  if (direct_source.find("ulong input_count;") == std::string::npos ||
      direct_source.find("ulong output_count;") == std::string::npos ||
      direct_source.find("ulong window_size;") == std::string::npos ||
      direct_source.find("ulong stride;") == std::string::npos ||
      direct_source.find("ulong padding;") == std::string::npos ||
      direct_source.find("if (input_index >= params.input_count)") ==
          std::string::npos ||
      direct_source.find("valid = false;") == std::string::npos ||
      direct_clamp_source.find("input_index = params.input_count - 1ul;") ==
          std::string::npos ||
      prefix_source.find("const ulong anchor = i * params.stride;") ==
          std::string::npos ||
      prefix_source.find("left_missing") != std::string::npos ||
      block_source.find("const ulong window = params.window_size;") ==
          std::string::npos ||
      block_source.find("const ulong left = i * params.stride;") ==
          std::string::npos ||
      block_source.find("2147483647") == std::string::npos ||
      saturating_source.find("device const int* input") == std::string::npos ||
      saturating_source.find("rund_range_add_sat(value, sample)") ==
          std::string::npos ||
      saturating_source.find("slot < params.window_size") ==
          std::string::npos) {
    return false;
  }
#if defined(RUND_NODE_HAVE_VULKAN_SDK)
  const RangePlan vulkan_direct = PlanRange(
      direct_clip.shape(), Gpu(RangeSource::Vulkan, kRangeWidth64Bit, 64u, 0u,
                               0u, std::numeric_limits<u32>::max(), direct));
  const RangePlan vulkan_block =
      PlanRange(block_clip.shape(),
                Gpu(RangeSource::Vulkan, kRangeWidth64Bit, 64u, 0u, 0u,
                    std::numeric_limits<u32>::max(), direct_block));
  const auto vulkan_direct_exec = RangeExec::from(vulkan_direct);
  const auto vulkan_exec = RangeExec::from(vulkan_block);
  if (!vulkan_direct_exec.has_value() || !vulkan_exec.has_value()) {
    return false;
  }
  const std::string direct_vulkan_source =
      VulkanRangeSource(*vulkan_direct_exec);
  const std::string source = VulkanRangeSource(*vulkan_exec);
  std::uint64_t bytes = 0u;
  if (!VulkanRangeSourceBytes(*vulkan_exec, bytes) || bytes != source.size() ||
      source.find("uint64_t output_count;") == std::string::npos ||
      source.find("const uint64_t left = block * params.stride;") ==
          std::string::npos ||
      direct_vulkan_source.find("if (input_index >= params.input_count)") ==
          std::string::npos ||
      direct_vulkan_source.find("valid = false;") == std::string::npos ||
      source.find("2147483647") == std::string::npos ||
      !VulkanRangeSourceMatches(*vulkan_exec, source, SourceHash(source))) {
    return false;
  }
#endif
  return true;
}

[[nodiscard]] constexpr bool FailClosedContract() {
  const auto invalid_operation = RangeTraits::make(
      static_cast<RangeOp>(255u), ComputeDomain::U32, RangeLaw::ModuloWidth);
  const auto invalid_domain =
      RangeTraits::sum_modulo(static_cast<ComputeDomain>(255u));
  const auto invalid_law = RangeTraits::make(
      RangeOp::Minimum, ComputeDomain::U32, RangeLaw::ModuloWidth);
  const auto invalid_shape = RangeShape::window(
      Traits(RangeOp::Sum), RangeBoundary::Clamp, 1u, 0u, 4u);
  const auto invalid_domain_width =
      RangeShape::window(Traits(RangeOp::Sum, ComputeDomain::U64),
                         RangeBoundary::Clamp, 1u, 1u, 4u);
  const auto payload_overflow =
      RangeShape::window(Traits(RangeOp::Sum), RangeBoundary::Clamp,
                         std::numeric_limits<u64>::max() / 4u + 1u, 1u, 4u);
  const auto invalid_caps =
      RangeCaps::gpu(RangeSource::Metal, 0x80u, 256u, 4u, 32768u, 1u,
                     std::numeric_limits<u64>::max(), kAllRangeCandidates);
  const RangePlan unavailable =
      PlanRange(Shape(RangeOp::Sum, 1u, 1u), RangeCaps::unavailable());
  constexpr u64 maximum_u32_count = std::numeric_limits<u64>::max() / 4u;
  constexpr std::uint8_t direct_block =
      RangeSupportBit(RangeSupport::Direct) |
      RangeSupportBit(RangeSupport::BlockPrefixSuffix);
  const RangePlan overflowing_block =
      PlanRange(Shape(RangeOp::Minimum, maximum_u32_count, maximum_u32_count),
                Gpu(RangeSource::Metal, kRangeWidth64Bit, 64u, 0u, 0u,
                    std::numeric_limits<u64>::max(), direct_block));
  constexpr u64 vulkan_input = std::numeric_limits<u32>::max() / 2u;
  constexpr u64 vulkan_window = 2u * vulkan_input + 1u;
  constexpr u64 vulkan_padding = vulkan_window - 1u;
  const RangeShape oversized_vulkan_span = AffineShape(
      RangeOp::Minimum, RangeBoundary::Clip, vulkan_input, 2u, vulkan_window,
      vulkan_input + vulkan_padding - 1u, vulkan_padding);
  const RangePlan vulkan_span_fallback =
      PlanRange(oversized_vulkan_span,
                Gpu(RangeSource::Vulkan, kRangeWidth64Bit, 64u, 0u, 0u,
                    std::numeric_limits<u32>::max(), direct_block));
  rund::kernel::u128 ignored = 0u;
  return !invalid_operation.has_value() && !invalid_domain.has_value() &&
         !invalid_law.has_value() && !invalid_shape.has_value() &&
         !invalid_domain_width.has_value() && !invalid_caps.has_value() &&
         !payload_overflow.has_value() && !unavailable.ok() &&
         std::string_view{unavailable.reason()} ==
             "compute_range_aggregate_unavailable" &&
         overflowing_block.ok() &&
         overflowing_block.candidate().disposition() == RangePath::Direct &&
         overflowing_block.legal_candidate_count() == 1u &&
         vulkan_span_fallback.ok() &&
         vulkan_span_fallback.candidate().disposition() == RangePath::Direct &&
         vulkan_span_fallback.legal_candidate_count() == 1u &&
         !range_plan_detail::Multiply(range_plan_detail::kU128Maximum, 2u,
                                      ignored);
}

static_assert(BasicSelectionContract());
static_assert(AlgebraContract());
static_assert(WidthAndCapacityContract());
static_assert(CandidateFamilyLegalityContract());
static_assert(CapabilityBoundaryContract());
static_assert(StorageIndexCapabilityContract());
static_assert(StorageBindingCapabilityContract());
static_assert(CostCrossoverContract());
static_assert(PrefixHierarchyContract());
static_assert(PrefixStageSubstrateContract());
static_assert(BlockPrefixSuffixContract());
static_assert(IdentityContract());
static_assert(LinearWorkContract());
static_assert(AffineShapeContract());
static_assert(AffineCostAndTopologyContract());
static_assert(AffinePaddedDirectContract());
static_assert(CpuCandidateContract());
static_assert(AffineIdentityContract());
static_assert(FailClosedContract());
static_assert(sizeof(RangePlan) <= 320u);

} // namespace

int RunRangePlannerContract() {
  return BasicSelectionContract() && AlgebraContract() &&
                 WidthAndCapacityContract() &&
                 CandidateFamilyLegalityContract() &&
                 ExhaustivePlannerContract() && CapabilityBoundaryContract() &&
                 StorageIndexCapabilityContract() &&
                 StorageBindingCapabilityContract() &&
                 CostCrossoverContract() && PrefixHierarchyContract() &&
                 PrefixStageSubstrateContract() &&
                 BlockPrefixSuffixContract() && IdentityContract() &&
                 ExecutionProjectionContract() && LinearWorkContract() &&
                 AffineShapeContract() && AffineCostAndTopologyContract() &&
                 AffinePaddedDirectContract() && CpuCandidateContract() &&
                 AffineIdentityContract() && AffineSourceContract() &&
                 FailClosedContract()
             ? 0
             : 1;
}

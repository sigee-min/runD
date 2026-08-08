#include "src/accel/range_aggregate/plan.hpp"

#include <kernel/program/compute/stencil/plan.hpp>

#include <array>
#include <cstdint>
#include <limits>
#include <optional>
#include <string_view>

namespace {

using rund::kernel::ComputeDomain;
using rund::kernel::u32;
using rund::kernel::u64;
using namespace rund::node::accel::detail;

inline constexpr std::uint8_t kAllRangeCandidates =
    RangeAggregateSupportBit(RangeAggregateSupport::Direct) |
    RangeAggregateSupportBit(RangeAggregateSupport::SharedHalo) |
    RangeAggregateSupportBit(RangeAggregateSupport::PrefixDifference) |
    RangeAggregateSupportBit(RangeAggregateSupport::BlockPrefixSuffix);

[[nodiscard]] constexpr RangeAggregateTraits
Traits(const RangeAggregateOperation operation,
       const ComputeDomain domain = ComputeDomain::U32) {
  const std::optional<RangeAggregateTraits> traits =
      RangeAggregateTraits::make(operation, domain,
                                 operation == RangeAggregateOperation::Sum
                                     ? RangeAggregateArithmeticLaw::ModuloWidth
                                     : RangeAggregateArithmeticLaw::OrderOnly);
  return *traits;
}

[[nodiscard]] constexpr RangeAggregateShape
Shape(const RangeAggregateOperation operation, const u64 count,
      const u64 radius, const u32 element_bytes = 4u,
      const ComputeDomain domain = ComputeDomain::U32) {
  return *RangeAggregateShape::window(Traits(operation, domain),
                                      RangeAggregateBoundary::Clamp, count,
                                      radius, element_bytes);
}

[[nodiscard]] constexpr RangeAggregateCapabilities
Gpu(const RangeAggregateSourceVariant variant,
    const std::uint8_t widths = kRangeAggregateKnownWidthMask,
    const u32 maximum_threads = 256u, const u32 occupancy = 4u,
    const u64 shared_limit = 32768u,
    const u64 maximum_groups = std::numeric_limits<u32>::max(),
    const std::uint8_t support = kAllRangeCandidates) {
  return *RangeAggregateCapabilities::gpu(variant, widths, maximum_threads,
                                          occupancy, shared_limit,
                                          maximum_groups, support);
}

[[nodiscard]] constexpr bool BasicSelectionContract() {
  const RangeAggregateCapabilities full =
      Gpu(RangeAggregateSourceVariant::Metal);
  const RangeAggregatePlan small_sum =
      PlanRangeAggregate(Shape(RangeAggregateOperation::Sum, 515u, 1u), full);
  const RangeAggregatePlan large_sum =
      PlanRangeAggregate(Shape(RangeAggregateOperation::Sum, 515u, 515u), full);
  const RangeAggregatePlan small_min = PlanRangeAggregate(
      Shape(RangeAggregateOperation::Minimum, 515u, 1u), full);
  const RangeAggregatePlan large_max = PlanRangeAggregate(
      Shape(RangeAggregateOperation::Maximum, 515u, 515u), full);
  const RangeAggregatePlan cpu =
      PlanRangeAggregate(Shape(RangeAggregateOperation::Sum, 17u, 3u),
                         RangeAggregateCapabilities::cpu());
  return small_sum.ok() &&
         small_sum.candidate().disposition() ==
             RangeAggregateCandidateDisposition::SharedHalo &&
         small_sum.candidate().width() == 256u && large_sum.ok() &&
         large_sum.candidate().disposition() ==
             RangeAggregateCandidateDisposition::PrefixDifference &&
         small_min.ok() &&
         small_min.candidate().disposition() ==
             RangeAggregateCandidateDisposition::SharedHalo &&
         large_max.ok() &&
         large_max.candidate().disposition() ==
             RangeAggregateCandidateDisposition::BlockPrefixSuffix &&
         cpu.ok() && cpu.candidate() == RangeAggregateCandidate::direct_cpu() &&
         cpu.stage_count() == 1u && cpu.temporary_count() == 0u;
}

[[nodiscard]] constexpr bool AlgebraAndProjectionContract() {
  constexpr std::array domains{ComputeDomain::I32, ComputeDomain::U32,
                               ComputeDomain::I64, ComputeDomain::U64,
                               ComputeDomain::Fixed};
  for (const ComputeDomain domain : domains) {
    const RangeAggregateTraits sum =
        Traits(RangeAggregateOperation::Sum, domain);
    const RangeAggregateTraits minimum =
        Traits(RangeAggregateOperation::Minimum, domain);
    const RangeAggregateTraits maximum =
        Traits(RangeAggregateOperation::Maximum, domain);
    const RangeAggregateTraits saturating =
        *RangeAggregateTraits::sum_saturating(domain);
    if (!sum.associative() || !sum.has_identity() || !sum.commutative() ||
        !sum.invertible() || sum.idempotent() || sum.ordered() ||
        !minimum.associative() || !minimum.has_identity() ||
        !minimum.commutative() || minimum.invertible() ||
        !minimum.idempotent() || !minimum.ordered() || maximum.invertible() ||
        !maximum.idempotent() || !maximum.ordered() ||
        saturating.associative() || saturating.invertible() ||
        !saturating.has_identity()) {
      return false;
    }
    const bool signed_domain = domain == ComputeDomain::I32 ||
                               domain == ComputeDomain::I64 ||
                               domain == ComputeDomain::Fixed;
    const bool unsigned_domain =
        domain == ComputeDomain::U32 || domain == ComputeDomain::U64;
    if (sum.signed_domain() != signed_domain ||
        sum.unsigned_domain() != unsigned_domain ||
        sum.fixed_domain() != (domain == ComputeDomain::Fixed)) {
      return false;
    }
  }

  constexpr rund::kernel::StencilDesc descriptor{
      .op = rund::kernel::StencilOp::Max,
      .element = rund::kernel::StencilElement::U64,
      .boundary = rund::kernel::StencilBoundary::Clamp,
      .element_count = 129u,
      .radius = 17u};
  constexpr rund::kernel::StencilPlan semantic =
      rund::kernel::PlanStencil(descriptor);
  constexpr auto projected =
      RangeAggregateShape::from_stencil(semantic, ComputeDomain::I64);
  return projected.has_value() &&
         projected->traits().operation() == RangeAggregateOperation::Maximum &&
         projected->traits().domain() == ComputeDomain::I64 &&
         projected->element_count() == 129u && projected->radius() == 17u &&
         projected->element_bytes() == 8u;
}

[[nodiscard]] constexpr bool WidthAndCapacityContract() {
  constexpr std::array<u32, 3u> widths{64u, 128u, 256u};
  constexpr std::array<std::uint8_t, 3u> width_bits{kRangeAggregateWidth64Bit,
                                                    kRangeAggregateWidth128Bit,
                                                    kRangeAggregateWidth256Bit};
  constexpr std::uint8_t direct_shared =
      RangeAggregateSupportBit(RangeAggregateSupport::Direct) |
      RangeAggregateSupportBit(RangeAggregateSupport::SharedHalo);
  for (std::size_t width_index = 0u; width_index < widths.size();
       ++width_index) {
    const u32 width = widths[width_index];
    const std::array<u32, 3u> capacities{9u, width - 1u, width};
    for (const u32 capacity : capacities) {
      for (const u32 element_bytes : std::array<u32, 2u>{4u, 8u}) {
        const u64 exact_limit =
            static_cast<u64>(width + 2u * capacity) * element_bytes * 4u;
        const RangeAggregateShape shape = Shape(
            RangeAggregateOperation::Sum, width + 1u, capacity, element_bytes,
            element_bytes == 4u ? ComputeDomain::U32 : ComputeDomain::U64);
        const RangeAggregatePlan below = PlanRangeAggregate(
            shape, Gpu(RangeAggregateSourceVariant::Metal,
                       width_bits[width_index], width, 4u, exact_limit - 1u,
                       std::numeric_limits<u32>::max(), direct_shared));
        const RangeAggregatePlan exact = PlanRangeAggregate(
            shape, Gpu(RangeAggregateSourceVariant::Metal,
                       width_bits[width_index], width, 4u, exact_limit,
                       std::numeric_limits<u32>::max(), direct_shared));
        const RangeAggregatePlan above = PlanRangeAggregate(
            shape, Gpu(RangeAggregateSourceVariant::Metal,
                       width_bits[width_index], width, 4u, exact_limit + 1u,
                       std::numeric_limits<u32>::max(), direct_shared));
        if (!below.ok() ||
            below.candidate().disposition() !=
                RangeAggregateCandidateDisposition::Direct ||
            !exact.ok() ||
            exact.candidate() !=
                *RangeAggregateCandidate::shared_halo(width, capacity) ||
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
      RangeAggregateSupportBit(RangeAggregateSupport::Direct) |
      RangeAggregateSupportBit(RangeAggregateSupport::PrefixDifference);
  constexpr std::uint8_t direct_block =
      RangeAggregateSupportBit(RangeAggregateSupport::Direct) |
      RangeAggregateSupportBit(RangeAggregateSupport::BlockPrefixSuffix);
  const RangeAggregateCapabilities prefix =
      Gpu(RangeAggregateSourceVariant::Metal, kRangeAggregateWidth64Bit, 64u,
          4u, 32768u, std::numeric_limits<u32>::max(), direct_prefix);
  const RangeAggregateCapabilities block =
      Gpu(RangeAggregateSourceVariant::Vulkan, kRangeAggregateWidth64Bit, 64u,
          0u, 0u, std::numeric_limits<u32>::max(), direct_block);
  const RangeAggregatePlan sum = PlanRangeAggregate(
      Shape(RangeAggregateOperation::Sum, 4097u, 4097u), prefix);
  const RangeAggregatePlan minimum_on_prefix = PlanRangeAggregate(
      Shape(RangeAggregateOperation::Minimum, 4097u, 4097u), prefix);
  const RangeAggregatePlan maximum = PlanRangeAggregate(
      Shape(RangeAggregateOperation::Maximum, 4097u, 4097u), block);
  const RangeAggregatePlan sum_on_block = PlanRangeAggregate(
      Shape(RangeAggregateOperation::Sum, 4097u, 4097u), block);
  const RangeAggregateTraits saturating =
      *RangeAggregateTraits::sum_saturating(ComputeDomain::Fixed);
  const RangeAggregatePlan saturating_sum = PlanRangeAggregate(
      *RangeAggregateShape::window(saturating, RangeAggregateBoundary::Clamp,
                                   4097u, 4097u, 4u),
      prefix);
  return sum.ok() &&
         sum.candidate().disposition() ==
             RangeAggregateCandidateDisposition::PrefixDifference &&
         minimum_on_prefix.ok() &&
         minimum_on_prefix.candidate().disposition() ==
             RangeAggregateCandidateDisposition::Direct &&
         maximum.ok() &&
         maximum.candidate().disposition() ==
             RangeAggregateCandidateDisposition::BlockPrefixSuffix &&
         sum_on_block.ok() &&
         sum_on_block.candidate().disposition() ==
             RangeAggregateCandidateDisposition::Direct &&
         saturating_sum.ok() &&
         saturating_sum.candidate().disposition() ==
             RangeAggregateCandidateDisposition::Direct;
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
  constexpr std::array operations{RangeAggregateOperation::Sum,
                                  RangeAggregateOperation::Minimum,
                                  RangeAggregateOperation::Maximum};
  constexpr std::array<u32, 3u> widths{64u, 128u, 256u};
  constexpr std::array<std::uint8_t, 3u> width_bits{kRangeAggregateWidth64Bit,
                                                    kRangeAggregateWidth128Bit,
                                                    kRangeAggregateWidth256Bit};

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
        const RangeAggregateCapabilities capabilities =
            Gpu(RangeAggregateSourceVariant::Metal, width_bits[width_index],
                width, 4u, shared_limit, groups, kAllRangeCandidates);
        const std::array<u64, 3u> radii{1u, count < 9u ? count : 9u, count};
        for (const u64 radius : radii) {
          for (const RangeAggregateOperation operation : operations) {
            const RangeAggregatePlan plan = PlanRangeAggregate(
                Shape(operation, count, radius, domain_width.bytes,
                      domain_width.domain),
                capabilities);
            const RangeAggregateCandidateDisposition expected =
                radius <= width ? RangeAggregateCandidateDisposition::SharedHalo
                : operation == RangeAggregateOperation::Sum
                    ? RangeAggregateCandidateDisposition::PrefixDifference
                    : RangeAggregateCandidateDisposition::BlockPrefixSuffix;
            const std::uint8_t expected_legal = radius <= width ? 3u : 2u;
            if (!plan.ok() || plan.candidate().disposition() != expected ||
                plan.candidate().width() != width ||
                (expected == RangeAggregateCandidateDisposition::SharedHalo &&
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
      RangeAggregateSupportBit(RangeAggregateSupport::Direct) |
      RangeAggregateSupportBit(RangeAggregateSupport::SharedHalo);
  constexpr u64 exact_limit = (64u + 2u * 9u) * 4u * 4u;
  const RangeAggregateShape shape =
      Shape(RangeAggregateOperation::Sum, 65u, 9u);
  const RangeAggregatePlan below = PlanRangeAggregate(
      shape, Gpu(RangeAggregateSourceVariant::Vulkan, kRangeAggregateWidth64Bit,
                 64u, 4u, exact_limit - 1u, std::numeric_limits<u32>::max(),
                 direct_shared));
  const RangeAggregatePlan exact = PlanRangeAggregate(
      shape,
      Gpu(RangeAggregateSourceVariant::Vulkan, kRangeAggregateWidth64Bit, 64u,
          4u, exact_limit, std::numeric_limits<u32>::max(), direct_shared));
  const RangeAggregateCapabilities direct_only =
      Gpu(RangeAggregateSourceVariant::Metal, kRangeAggregateWidth64Bit, 64u,
          0u, 0u, 2u, RangeAggregateSupportBit(RangeAggregateSupport::Direct));
  const RangeAggregateCapabilities one_group =
      Gpu(RangeAggregateSourceVariant::Metal, kRangeAggregateWidth64Bit, 64u,
          0u, 0u, 1u, RangeAggregateSupportBit(RangeAggregateSupport::Direct));
  const RangeAggregateCapabilities three_groups =
      Gpu(RangeAggregateSourceVariant::Metal, kRangeAggregateWidth64Bit, 64u,
          0u, 0u, 3u, RangeAggregateSupportBit(RangeAggregateSupport::Direct));
  const RangeAggregatePlan dispatch_below = PlanRangeAggregate(
      Shape(RangeAggregateOperation::Sum, 128u, 1u), one_group);
  const RangeAggregatePlan dispatch_exact = PlanRangeAggregate(
      Shape(RangeAggregateOperation::Sum, 128u, 1u), direct_only);
  const RangeAggregatePlan dispatch_above = PlanRangeAggregate(
      Shape(RangeAggregateOperation::Sum, 128u, 1u), three_groups);
  const RangeAggregatePlan dispatch_over = PlanRangeAggregate(
      Shape(RangeAggregateOperation::Sum, 129u, 1u), direct_only);
  return below.ok() &&
         below.candidate().disposition() ==
             RangeAggregateCandidateDisposition::Direct &&
         exact.ok() &&
         exact.candidate() == *RangeAggregateCandidate::shared_halo(64u, 9u) &&
         exact.cost().shared_bytes == (64u + 18u) * 4u &&
         !dispatch_below.ok() && dispatch_exact.ok() && dispatch_above.ok() &&
         !dispatch_over.ok() &&
         std::string_view{dispatch_over.reason()} ==
             "compute_range_aggregate_candidate_unavailable";
}

[[nodiscard]] constexpr bool CostCrossoverContract() {
  constexpr std::uint8_t direct_prefix =
      RangeAggregateSupportBit(RangeAggregateSupport::Direct) |
      RangeAggregateSupportBit(RangeAggregateSupport::PrefixDifference);
  constexpr std::uint8_t direct_block =
      RangeAggregateSupportBit(RangeAggregateSupport::Direct) |
      RangeAggregateSupportBit(RangeAggregateSupport::BlockPrefixSuffix);
  const RangeAggregateCapabilities prefix =
      Gpu(RangeAggregateSourceVariant::Metal, kRangeAggregateWidth64Bit, 64u,
          4u, 32768u, std::numeric_limits<u32>::max(), direct_prefix);
  const RangeAggregateCapabilities block =
      Gpu(RangeAggregateSourceVariant::Vulkan, kRangeAggregateWidth64Bit, 64u,
          0u, 0u, std::numeric_limits<u32>::max(), direct_block);
  const RangeAggregatePlan sum_before =
      PlanRangeAggregate(Shape(RangeAggregateOperation::Sum, 515u, 1u), prefix);
  const RangeAggregatePlan sum_after =
      PlanRangeAggregate(Shape(RangeAggregateOperation::Sum, 515u, 2u), prefix);
  const RangeAggregatePlan minimum_before = PlanRangeAggregate(
      Shape(RangeAggregateOperation::Minimum, 515u, 1u), block);
  const RangeAggregatePlan minimum_after = PlanRangeAggregate(
      Shape(RangeAggregateOperation::Minimum, 515u, 2u), block);
  return sum_before.ok() && sum_after.ok() && minimum_before.ok() &&
         minimum_after.ok() &&
         sum_before.candidate().disposition() ==
             RangeAggregateCandidateDisposition::Direct &&
         sum_after.candidate().disposition() ==
             RangeAggregateCandidateDisposition::PrefixDifference &&
         minimum_before.candidate().disposition() ==
             RangeAggregateCandidateDisposition::Direct &&
         minimum_after.candidate().disposition() ==
             RangeAggregateCandidateDisposition::BlockPrefixSuffix;
}

[[nodiscard]] constexpr bool PrefixHierarchyContract() {
  constexpr std::uint8_t direct_prefix =
      RangeAggregateSupportBit(RangeAggregateSupport::Direct) |
      RangeAggregateSupportBit(RangeAggregateSupport::PrefixDifference);
  const RangeAggregateCapabilities capabilities =
      Gpu(RangeAggregateSourceVariant::Metal, kRangeAggregateWidth64Bit, 64u,
          4u, 32768u, std::numeric_limits<u32>::max(), direct_prefix);
  const RangeAggregatePlan plan = PlanRangeAggregate(
      Shape(RangeAggregateOperation::Sum, 4097u, 4097u), capabilities);
  constexpr std::array expected{
      RangeAggregateStageDisposition::PrefixBlock,
      RangeAggregateStageDisposition::PrefixSummary,
      RangeAggregateStageDisposition::PrefixSummary,
      RangeAggregateStageDisposition::PrefixFixup,
      RangeAggregateStageDisposition::PrefixFixup,
      RangeAggregateStageDisposition::PrefixWindow,
  };
  if (!plan.ok() ||
      plan.candidate().disposition() !=
          RangeAggregateCandidateDisposition::PrefixDifference ||
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
  return plan.temporary(0u).role == RangeTemporaryRole::PrefixValues &&
         plan.temporary(0u).first_stage == 0u &&
         plan.temporary(0u).last_stage == 5u &&
         plan.temporary(1u).role == RangeTemporaryRole::BlockSummaries &&
         plan.temporary(1u).bytes == 65u * 4u &&
         plan.temporary(1u).last_stage == 4u &&
         plan.temporary(2u).bytes == 2u * 4u &&
         plan.temporary(2u).last_stage == 3u;
}

[[nodiscard]] constexpr bool PrefixStageSubstrateContract() {
  constexpr auto hierarchy = PlanRangeAggregatePrefixHierarchy(
      4097u, 64u, 4u, std::numeric_limits<u32>::max());
  constexpr auto flat_single = PlanRangeAggregateFlatPrefix(256u, 1u, 128u, 4u);
  constexpr auto flat_multiple =
      PlanRangeAggregateFlatPrefix(513u, 3u, 128u, 8u);
  constexpr auto invalid = PlanRangeAggregateFlatPrefix(0u, 0u, 128u, 4u);
  constexpr auto dispatch_limited =
      PlanRangeAggregatePrefixHierarchy(4097u, 64u, 4u, 64u);
  return hierarchy.ok() &&
         hierarchy.disposition() ==
             RangeAggregatePrefixDisposition::Hierarchical &&
         hierarchy.stage_count() == 5u && hierarchy.temporary_count() == 2u &&
         hierarchy.stage(0u) ==
             RangeAggregateStagePlan{
                 .disposition = RangeAggregateStageDisposition::PrefixBlock,
                 .level = 0u,
                 .element_count = 4097u,
                 .groups = 65u,
                 .width = 64u} &&
         hierarchy.stage(4u) ==
             RangeAggregateStagePlan{
                 .disposition = RangeAggregateStageDisposition::PrefixFixup,
                 .level = 0u,
                 .element_count = 4097u,
                 .groups = 65u,
                 .width = 64u} &&
         hierarchy.temporary(0u).bytes == 260u &&
         hierarchy.temporary(0u).last_stage == 4u &&
         hierarchy.temporary(1u).bytes == 8u &&
         hierarchy.temporary(1u).last_stage == 3u && flat_single.ok() &&
         flat_single.disposition() ==
             RangeAggregatePrefixDisposition::FlatBlockTotals &&
         flat_single.stage_count() == 1u &&
         flat_single.temporary_count() == 1u &&
         flat_single.stage(0u).groups == 1u &&
         flat_single.temporary(0u).bytes == 4u &&
         flat_single.temporary(0u).last_stage == 0u && flat_multiple.ok() &&
         flat_multiple.disposition() ==
             RangeAggregatePrefixDisposition::FlatBlockTotals &&
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
      RangeAggregateSupportBit(RangeAggregateSupport::Direct) |
      RangeAggregateSupportBit(RangeAggregateSupport::BlockPrefixSuffix);
  const RangeAggregateCapabilities capabilities =
      Gpu(RangeAggregateSourceVariant::Vulkan, kRangeAggregateWidth64Bit, 64u,
          0u, 0u, std::numeric_limits<u32>::max(), direct_block);
  const RangeAggregatePlan plan = PlanRangeAggregate(
      Shape(RangeAggregateOperation::Minimum, 4097u, 4097u), capabilities);
  return plan.ok() &&
         plan.candidate().disposition() ==
             RangeAggregateCandidateDisposition::BlockPrefixSuffix &&
         plan.stage_count() == 2u && plan.temporary_count() == 2u &&
         plan.stage(0u) ==
             RangeAggregateStagePlan{
                 .disposition =
                     RangeAggregateStageDisposition::BlockPrefixSuffix,
                 .level = 0u,
                 .element_count = 12291u,
                 .groups = 1u,
                 .width = 64u} &&
         plan.stage(1u).disposition ==
             RangeAggregateStageDisposition::BlockWindow &&
         plan.stage(1u).groups == 65u && plan.cost().scratch_bytes == 98328u &&
         plan.cost().dispatch_count == 2u;
}

[[nodiscard]] constexpr bool IdentityContract() {
  constexpr std::uint8_t direct_prefix =
      RangeAggregateSupportBit(RangeAggregateSupport::Direct) |
      RangeAggregateSupportBit(RangeAggregateSupport::PrefixDifference);
  const RangeAggregateCapabilities metal =
      Gpu(RangeAggregateSourceVariant::Metal, kRangeAggregateWidth256Bit, 256u,
          4u, 32768u, std::numeric_limits<u32>::max(), direct_prefix);
  const RangeAggregateCapabilities vulkan =
      Gpu(RangeAggregateSourceVariant::Vulkan, kRangeAggregateWidth256Bit, 256u,
          4u, 32768u, std::numeric_limits<u32>::max(), direct_prefix);
  const RangeAggregatePlan first = PlanRangeAggregate(
      Shape(RangeAggregateOperation::Sum, 515u, 300u), metal);
  const RangeAggregatePlan second = PlanRangeAggregate(
      Shape(RangeAggregateOperation::Sum, 515u, 515u), metal);
  const RangeAggregatePlan other_backend = PlanRangeAggregate(
      Shape(RangeAggregateOperation::Sum, 515u, 300u), vulkan);
  return first.ok() && second.ok() && other_backend.ok() &&
         first.candidate() == second.candidate() &&
         first.source_identity() == second.source_identity() &&
         first.execution_identity() != second.execution_identity() &&
         first.source_identity() != other_backend.source_identity();
}

[[nodiscard]] constexpr bool LinearWorkContract() {
  constexpr u64 count = 4097u;
  constexpr std::uint8_t direct_prefix =
      RangeAggregateSupportBit(RangeAggregateSupport::Direct) |
      RangeAggregateSupportBit(RangeAggregateSupport::PrefixDifference);
  constexpr std::uint8_t direct_block =
      RangeAggregateSupportBit(RangeAggregateSupport::Direct) |
      RangeAggregateSupportBit(RangeAggregateSupport::BlockPrefixSuffix);
  const RangeAggregateCapabilities prefix_capabilities =
      Gpu(RangeAggregateSourceVariant::Metal, kRangeAggregateWidth64Bit, 64u,
          4u, 32768u, std::numeric_limits<u32>::max(), direct_prefix);
  const RangeAggregateCapabilities block_capabilities =
      Gpu(RangeAggregateSourceVariant::Vulkan, kRangeAggregateWidth64Bit, 64u,
          0u, 0u, std::numeric_limits<u32>::max(), direct_block);
  const RangeAggregatePlan prefix_half =
      PlanRangeAggregate(Shape(RangeAggregateOperation::Sum, count, count / 2u),
                         prefix_capabilities);
  const RangeAggregatePlan prefix_full = PlanRangeAggregate(
      Shape(RangeAggregateOperation::Sum, count, count), prefix_capabilities);
  const RangeAggregatePlan block_half = PlanRangeAggregate(
      Shape(RangeAggregateOperation::Minimum, count, count / 2u),
      block_capabilities);
  const RangeAggregatePlan block_full =
      PlanRangeAggregate(Shape(RangeAggregateOperation::Minimum, count, count),
                         block_capabilities);
  return prefix_half.ok() && prefix_full.ok() && block_half.ok() &&
         block_full.ok() &&
         prefix_half.candidate().disposition() ==
             RangeAggregateCandidateDisposition::PrefixDifference &&
         prefix_full.candidate() == prefix_half.candidate() &&
         prefix_full.stage_count() == prefix_half.stage_count() &&
         prefix_full.temporary_count() == prefix_half.temporary_count() &&
         prefix_full.cost().scratch_bytes == prefix_half.cost().scratch_bytes &&
         prefix_full.cost().combine_ops <=
             static_cast<rund::kernel::u128>(8u) * count &&
         block_half.candidate().disposition() ==
             RangeAggregateCandidateDisposition::BlockPrefixSuffix &&
         block_full.candidate() == block_half.candidate() &&
         block_full.stage_count() == 2u &&
         block_full.cost().combine_ops <=
             static_cast<rund::kernel::u128>(7u) * count &&
         block_full.cost().scratch_bytes <= 6u * count * 4u;
}

[[nodiscard]] constexpr bool FailClosedContract() {
  const auto invalid_operation = RangeAggregateTraits::make(
      static_cast<RangeAggregateOperation>(255u), ComputeDomain::U32,
      RangeAggregateArithmeticLaw::ModuloWidth);
  const auto invalid_domain =
      RangeAggregateTraits::sum_modulo(static_cast<ComputeDomain>(255u));
  const auto invalid_law = RangeAggregateTraits::make(
      RangeAggregateOperation::Minimum, ComputeDomain::U32,
      RangeAggregateArithmeticLaw::ModuloWidth);
  const auto invalid_shape =
      RangeAggregateShape::window(Traits(RangeAggregateOperation::Sum),
                                  RangeAggregateBoundary::Clamp, 1u, 0u, 4u);
  const auto invalid_domain_width = RangeAggregateShape::window(
      Traits(RangeAggregateOperation::Sum, ComputeDomain::U64),
      RangeAggregateBoundary::Clamp, 1u, 1u, 4u);
  const auto payload_overflow = RangeAggregateShape::window(
      Traits(RangeAggregateOperation::Sum), RangeAggregateBoundary::Clamp,
      std::numeric_limits<u64>::max() / 4u + 1u, 1u, 4u);
  const auto invalid_caps = RangeAggregateCapabilities::gpu(
      RangeAggregateSourceVariant::Metal, 0x80u, 256u, 4u, 32768u, 1u,
      kAllRangeCandidates);
  const RangeAggregatePlan unavailable =
      PlanRangeAggregate(Shape(RangeAggregateOperation::Sum, 1u, 1u),
                         RangeAggregateCapabilities::unavailable());
  constexpr u64 maximum_u32_count = std::numeric_limits<u64>::max() / 4u;
  constexpr std::uint8_t direct_block =
      RangeAggregateSupportBit(RangeAggregateSupport::Direct) |
      RangeAggregateSupportBit(RangeAggregateSupport::BlockPrefixSuffix);
  const RangeAggregatePlan overflowing_block = PlanRangeAggregate(
      Shape(RangeAggregateOperation::Minimum, maximum_u32_count,
            maximum_u32_count),
      Gpu(RangeAggregateSourceVariant::Vulkan, kRangeAggregateWidth64Bit, 64u,
          0u, 0u, std::numeric_limits<u64>::max(), direct_block));
  rund::kernel::u128 ignored = 0u;
  return !invalid_operation.has_value() && !invalid_domain.has_value() &&
         !invalid_law.has_value() && !invalid_shape.has_value() &&
         !invalid_domain_width.has_value() && !invalid_caps.has_value() &&
         !payload_overflow.has_value() && !unavailable.ok() &&
         std::string_view{unavailable.reason()} ==
             "compute_range_aggregate_unavailable" &&
         overflowing_block.ok() &&
         overflowing_block.candidate().disposition() ==
             RangeAggregateCandidateDisposition::Direct &&
         overflowing_block.legal_candidate_count() == 1u &&
         !range_aggregate_plan_detail::Multiply(
             range_aggregate_plan_detail::kU128Maximum, 2u, ignored);
}

static_assert(BasicSelectionContract());
static_assert(AlgebraAndProjectionContract());
static_assert(WidthAndCapacityContract());
static_assert(CandidateFamilyLegalityContract());
static_assert(CapabilityBoundaryContract());
static_assert(CostCrossoverContract());
static_assert(PrefixHierarchyContract());
static_assert(PrefixStageSubstrateContract());
static_assert(BlockPrefixSuffixContract());
static_assert(IdentityContract());
static_assert(LinearWorkContract());
static_assert(FailClosedContract());
static_assert(sizeof(RangeAggregatePlan) <= 320u);

} // namespace

int RunAccelRangeAggregatePlannerContract() {
  return BasicSelectionContract() && AlgebraAndProjectionContract() &&
                 WidthAndCapacityContract() &&
                 CandidateFamilyLegalityContract() &&
                 ExhaustivePlannerContract() && CapabilityBoundaryContract() &&
                 CostCrossoverContract() && PrefixHierarchyContract() &&
                 PrefixStageSubstrateContract() &&
                 BlockPrefixSuffixContract() && IdentityContract() &&
                 LinearWorkContract() && FailClosedContract()
             ? 0
             : 1;
}

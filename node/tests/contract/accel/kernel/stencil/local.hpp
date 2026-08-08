#pragma once

#include <accel/device.hpp>

#include "../primitive/local.hpp"
#include "src/accel/range_aggregate/plan.hpp"
#include "src/accel/stencil/shape.hpp"

#include <limits>
#include <optional>

namespace node_accel_contract::stencil {

// Source contracts deliberately construct the same frozen RangeAggregatePlan
// consumed by the backend.  The support mask describes the legal source path;
// it is not a second candidate selector.
enum class SourcePlanPath : std::uint8_t {
  Direct,
  SharedHalo,
  PrefixDifference,
  BlockPrefixSuffix,
};

[[nodiscard]] constexpr std::uint8_t
SourcePlanWidthMask(const rund::kernel::u32 width) noexcept {
  using namespace rund::node::accel::detail;
  return width == 64u    ? kRangeAggregateWidth64Bit
         : width == 128u ? kRangeAggregateWidth128Bit
         : width == 256u ? kRangeAggregateWidth256Bit
                         : 0u;
}

[[nodiscard]] constexpr rund::node::accel::detail::RangeAggregatePlan
PlanStencilSourceVariant(
    const rund::node::accel::detail::RangeAggregateSourceVariant backend,
    const rund::kernel::StencilOp operation,
    const rund::kernel::ComputeDomain domain,
    const rund::node::accel::detail::StencilGpuShape physical_shape,
    const SourcePlanPath path) noexcept {
  using namespace rund::node::accel::detail;
  const std::optional<RangeAggregateTraits> traits =
      operation == rund::kernel::StencilOp::Sum
          ? RangeAggregateTraits::sum_modulo(domain)
      : operation == rund::kernel::StencilOp::Min
          ? RangeAggregateTraits::minimum(domain)
      : operation == rund::kernel::StencilOp::Max
          ? RangeAggregateTraits::maximum(domain)
          : std::nullopt;
  if (!traits.has_value() || !physical_shape.valid()) {
    return RangeAggregatePlan::rejected(
        "compute_range_aggregate_shape_invalid");
  }
  const rund::kernel::u64 radius =
      path == SourcePlanPath::SharedHalo
          ? physical_shape.radius_cap()
          : static_cast<rund::kernel::u64>(physical_shape.width());
  const rund::kernel::u64 count =
      static_cast<rund::kernel::u64>(physical_shape.width()) * 3u + 3u;
  const std::optional<RangeAggregateShape> shape = RangeAggregateShape::window(
      *traits, RangeAggregateBoundary::Clamp, count, radius,
      traits->domain() == rund::kernel::ComputeDomain::I64 ||
              traits->domain() == rund::kernel::ComputeDomain::U64 ||
              traits->domain() == rund::kernel::ComputeDomain::Fixed
          ? 8u
          : 4u);
  if (!shape.has_value()) {
    return RangeAggregatePlan::rejected(
        "compute_range_aggregate_shape_invalid");
  }
  const std::uint8_t direct =
      RangeAggregateSupportBit(RangeAggregateSupport::Direct);
  const std::uint8_t support =
      path == SourcePlanPath::Direct ? direct
      : path == SourcePlanPath::SharedHalo
          ? static_cast<std::uint8_t>(
                direct |
                RangeAggregateSupportBit(RangeAggregateSupport::SharedHalo))
      : path == SourcePlanPath::PrefixDifference
          ? static_cast<std::uint8_t>(
                direct | RangeAggregateSupportBit(
                             RangeAggregateSupport::PrefixDifference))
          : static_cast<std::uint8_t>(
                direct | RangeAggregateSupportBit(
                             RangeAggregateSupport::BlockPrefixSuffix));
  const rund::kernel::u64 shared_bytes =
      path == SourcePlanPath::SharedHalo
          ? static_cast<rund::kernel::u64>(physical_shape.width() +
                                           2u * physical_shape.radius_cap()) *
                shape->element_bytes()
      : path == SourcePlanPath::PrefixDifference
          ? static_cast<rund::kernel::u64>(physical_shape.width()) *
                shape->element_bytes()
          : 0u;
  const std::optional<RangeAggregateCapabilities> capabilities =
      RangeAggregateCapabilities::gpu(
          backend, SourcePlanWidthMask(physical_shape.width()),
          physical_shape.width(),
          shared_bytes == 0u ? 0u : kRangeAggregateSharedMemoryReserve,
          shared_bytes == 0u
              ? 0u
              : shared_bytes * kRangeAggregateSharedMemoryReserve,
          std::numeric_limits<rund::kernel::u32>::max(), support);
  if (!capabilities.has_value()) {
    return RangeAggregatePlan::rejected(
        "compute_range_aggregate_capabilities_invalid");
  }
  const RangeAggregatePlan plan = PlanRangeAggregate(*shape, *capabilities);
  const RangeAggregateCandidateDisposition expected =
      path == SourcePlanPath::Direct
          ? RangeAggregateCandidateDisposition::Direct
      : path == SourcePlanPath::SharedHalo
          ? RangeAggregateCandidateDisposition::SharedHalo
      : path == SourcePlanPath::PrefixDifference
          ? RangeAggregateCandidateDisposition::PrefixDifference
          : RangeAggregateCandidateDisposition::BlockPrefixSuffix;
  return plan.ok() && plan.candidate().disposition() == expected
             ? plan
             : RangeAggregatePlan::rejected(
                   "compute_range_aggregate_candidate_unavailable");
}

[[nodiscard]] bool MatchesU32(const rund::AccelDevice &pick);
[[nodiscard]] bool MatchesSumI32(const rund::AccelDevice &pick);
[[nodiscard]] bool MatchesU64(const rund::AccelDevice &pick);
[[nodiscard]] bool MatchesWideWindowU32(const rund::AccelDevice &pick);
[[nodiscard]] bool MatchesCount65U32(const rund::AccelDevice &pick);
[[nodiscard]] bool MatchesRadius64BoundaryU32(const rund::AccelDevice &pick);
[[nodiscard]] bool
MatchesCapabilitySharedBoundaryU32(const rund::AccelDevice &pick,
                                   rund::kernel::u64 radius);
[[nodiscard]] bool MatchesPrefixDifferenceU32(const rund::AccelDevice &pick);
[[nodiscard]] bool
MatchesForcedPrefixDifferenceU32(const rund::AccelDevice &pick);
[[nodiscard]] bool
MatchesForcedPrefixDifferenceU64(const rund::AccelDevice &pick);
[[nodiscard]] bool MatchesDeepPrefixHierarchyU32(const rund::AccelDevice &pick);
[[nodiscard]] bool MatchesMinU32(const rund::AccelDevice &pick);
[[nodiscard]] bool MatchesMinI32(const rund::AccelDevice &pick);
[[nodiscard]] bool MatchesMaxU64(const rund::AccelDevice &pick);
[[nodiscard]] bool
MatchesBlockPrefixSuffixMinI32(const rund::AccelDevice &pick);
[[nodiscard]] bool
MatchesBlockPrefixSuffixMaxI32(const rund::AccelDevice &pick);
[[nodiscard]] bool
MatchesBlockPrefixSuffixMinU64(const rund::AccelDevice &pick);
[[nodiscard]] bool
MatchesBlockPrefixSuffixMaxU64(const rund::AccelDevice &pick);
[[nodiscard]] bool
MatchesForcedBlockPrefixSuffixMinI32(const rund::AccelDevice &pick);
[[nodiscard]] bool
MatchesForcedBlockPrefixSuffixMaxI32(const rund::AccelDevice &pick);
[[nodiscard]] bool
MatchesForcedBlockPrefixSuffixMinU64(const rund::AccelDevice &pick);
[[nodiscard]] bool
MatchesForcedBlockPrefixSuffixMaxU64(const rund::AccelDevice &pick);

} // namespace node_accel_contract::stencil

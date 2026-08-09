#pragma once

#include <accel/device.hpp>

#include "../primitive/local.hpp"
#include "src/accel/range_aggregate/plan.hpp"
#include "src/accel/stencil/shape.hpp"

#include <cassert>
#include <limits>
#include <optional>

namespace node_accel_contract::stencil {

// Source contracts deliberately construct the same frozen RangePlan
// consumed by the backend.  The support mask describes the legal source path;
// it is not a second candidate selector.
enum class SourcePlanPath : std::uint8_t {
  Direct,
  SharedHalo,
  PrefixDifference,
  BlockPrefixSuffix,
};

// Contract fixtures use only the three concrete widths admitted by the
// execution contract.  Production code keeps factory failure explicit; this
// helper makes those already-validated fixture literals concise without
// reintroducing a second shape representation.
[[nodiscard]] constexpr rund::node::accel::detail::RangeGpuShape
RangeDirectShape(const rund::kernel::u32 width) noexcept {
  using namespace rund::node::accel::detail;
  const std::optional<RangeGpuShape> shape = RangeGpuShape::direct(width);
  assert(shape.has_value());
  return *shape;
}

[[nodiscard]] constexpr rund::node::accel::detail::RangeGpuShape
RangeSharedShape(const rund::kernel::u32 width,
                 const rund::kernel::u32 shared_radius_capacity) noexcept {
  using namespace rund::node::accel::detail;
  const std::optional<RangeGpuShape> shape =
      RangeGpuShape::shared_halo(width, shared_radius_capacity);
  assert(shape.has_value());
  return *shape;
}

[[nodiscard]] constexpr rund::node::accel::detail::RangeGpuShape
RequireRangeShape(const rund::node::accel::detail::RangePlan &plan) noexcept {
  using namespace rund::node::accel::detail;
  const std::optional<RangeGpuShape> shape = RangeGpuShapeFor(plan);
  assert(shape.has_value());
  return *shape;
}

[[nodiscard]] constexpr rund::node::accel::detail::RangeExec
RequireRangeExec(const rund::node::accel::detail::RangePlan &plan) noexcept {
  using namespace rund::node::accel::detail;
  const std::optional<RangeExec> execution = RangeExec::from(plan);
  assert(execution.has_value());
  return *execution;
}

[[nodiscard]] constexpr std::uint8_t
SourcePlanWidthMask(const rund::kernel::u32 width) noexcept {
  using namespace rund::node::accel::detail;
  return width == 64u    ? kRangeWidth64Bit
         : width == 128u ? kRangeWidth128Bit
         : width == 256u ? kRangeWidth256Bit
                         : 0u;
}

[[nodiscard]] constexpr rund::node::accel::detail::RangePlan
PlanStencilSourceVariant(
    const rund::node::accel::detail::RangeSource backend,
    const rund::kernel::StencilOp operation,
    const rund::kernel::ComputeDomain domain,
    const rund::node::accel::detail::RangeGpuShape physical_shape,
    const SourcePlanPath path) noexcept {
  using namespace rund::node::accel::detail;
  const std::optional<RangeTraits> traits =
      operation == rund::kernel::StencilOp::Sum
          ? RangeTraits::sum_modulo(domain)
      : operation == rund::kernel::StencilOp::Min ? RangeTraits::minimum(domain)
      : operation == rund::kernel::StencilOp::Max ? RangeTraits::maximum(domain)
                                                  : std::nullopt;
  if (!traits.has_value()) {
    return RangePlan::rejected("compute_range_aggregate_shape_invalid");
  }
  const rund::kernel::u64 radius =
      path == SourcePlanPath::SharedHalo
          ? physical_shape.shared_radius_capacity()
          : static_cast<rund::kernel::u64>(physical_shape.width());
  const rund::kernel::u64 count =
      static_cast<rund::kernel::u64>(physical_shape.width()) * 3u + 3u;
  const std::optional<RangeShape> shape = RangeShape::window(
      *traits, RangeBoundary::Clamp, count, radius,
      traits->domain() == rund::kernel::ComputeDomain::I64 ||
              traits->domain() == rund::kernel::ComputeDomain::U64 ||
              traits->domain() == rund::kernel::ComputeDomain::Fixed
          ? 8u
          : 4u);
  if (!shape.has_value()) {
    return RangePlan::rejected("compute_range_aggregate_shape_invalid");
  }
  const std::uint8_t direct = RangeSupportBit(RangeSupport::Direct);
  const std::uint8_t support =
      path == SourcePlanPath::Direct ? direct
      : path == SourcePlanPath::SharedHalo
          ? static_cast<std::uint8_t>(direct |
                                      RangeSupportBit(RangeSupport::SharedHalo))
      : path == SourcePlanPath::PrefixDifference
          ? static_cast<std::uint8_t>(
                direct | RangeSupportBit(RangeSupport::PrefixDifference))
          : static_cast<std::uint8_t>(
                direct | RangeSupportBit(RangeSupport::BlockPrefixSuffix));
  const rund::kernel::u64 shared_bytes =
      path == SourcePlanPath::SharedHalo
          ? static_cast<rund::kernel::u64>(
                physical_shape.width() +
                2u * physical_shape.shared_radius_capacity()) *
                shape->element_bytes()
      : path == SourcePlanPath::PrefixDifference
          ? static_cast<rund::kernel::u64>(physical_shape.width()) *
                shape->element_bytes()
          : 0u;
  const std::optional<RangeCaps> capabilities = RangeCaps::gpu(
      backend, SourcePlanWidthMask(physical_shape.width()),
      physical_shape.width(), shared_bytes == 0u ? 0u : kRangeSharedReserve,
      shared_bytes == 0u ? 0u : shared_bytes * kRangeSharedReserve,
      std::numeric_limits<rund::kernel::u32>::max(),
      backend == RangeSource::Vulkan
          ? std::numeric_limits<rund::kernel::u32>::max()
          : std::numeric_limits<rund::kernel::u64>::max(),
      support);
  if (!capabilities.has_value()) {
    return RangePlan::rejected("compute_range_aggregate_capabilities_invalid");
  }
  const RangePlan plan = PlanRange(*shape, *capabilities);
  const RangePath expected =
      path == SourcePlanPath::Direct             ? RangePath::Direct
      : path == SourcePlanPath::SharedHalo       ? RangePath::SharedHalo
      : path == SourcePlanPath::PrefixDifference ? RangePath::PrefixDifference
                                                 : RangePath::BlockPrefixSuffix;
  return plan.ok() && plan.candidate().disposition() == expected
             ? plan
             : RangePlan::rejected(
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

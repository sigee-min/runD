#pragma once

#include "src/accel/range_aggregate/execution/control.hpp"
#include "src/accel/range_aggregate/execution/projection.hpp"
#include "src/accel/range_aggregate/execution/run.hpp"
#include "src/accel/range_aggregate/plan.hpp"
#include "src/accel/range_aggregate/plan/build.hpp"

#include <cassert>
#include <limits>
#include <optional>

namespace node_accel_contract::range {

// Static checks evaluate the single production implementation. Runtime checks
// exercise the compiled entry point, so both execution forms retain coverage.
[[nodiscard]] constexpr rund::node::accel::detail::RangePlan
ContractPlanRange(const rund::node::accel::detail::RangeShape &shape,
                  const rund::node::accel::detail::RangeCaps &caps) noexcept {
  return std::is_constant_evaluated()
             ? rund::node::accel::detail::BuildRangePlan(shape, caps)
             : rund::node::accel::detail::PlanRange(shape, caps);
}

inline constexpr std::uint8_t kAllCandidates =
    rund::node::accel::detail::kRangeKnownSupportMask;

[[nodiscard]] constexpr std::optional<rund::node::accel::detail::RangeTraits>
MaybeTraits(const rund::node::accel::detail::RangeOp operation,
            const rund::kernel::ComputeDomain domain) noexcept {
  using namespace rund::node::accel::detail;
  switch (operation) {
  case RangeOp::Sum:
    return RangeTraits::sum_modulo(domain);
  case RangeOp::Minimum:
    return RangeTraits::minimum(domain);
  case RangeOp::Maximum:
    return RangeTraits::maximum(domain);
  }
  return std::nullopt;
}

[[nodiscard]] constexpr rund::node::accel::detail::RangeTraits
Traits(const rund::node::accel::detail::RangeOp operation,
       const rund::kernel::ComputeDomain domain =
           rund::kernel::ComputeDomain::U32) noexcept {
  const std::optional<rund::node::accel::detail::RangeTraits> traits =
      MaybeTraits(operation, domain);
  assert(traits.has_value());
  return *traits;
}

[[nodiscard]] constexpr rund::node::accel::detail::RangeShape
Shape(const rund::node::accel::detail::RangeOp operation,
      const rund::kernel::u64 count, const rund::kernel::u64 radius,
      const rund::kernel::u32 element_bytes = 4u,
      const rund::kernel::ComputeDomain domain =
          rund::kernel::ComputeDomain::U32) noexcept {
  using namespace rund::node::accel::detail;
  const std::optional<RangeShape> shape =
      RangeShape::affine(Traits(operation, domain), RangeBoundary::Clamp, count,
                         count, 2u * radius + 1u, 1u, radius, element_bytes);
  assert(shape.has_value());
  return *shape;
}

[[nodiscard]] constexpr rund::node::accel::detail::RangeShape AffineShape(
    const rund::node::accel::detail::RangeOp operation,
    const rund::node::accel::detail::RangeBoundary boundary,
    const rund::kernel::u64 input_count, const rund::kernel::u64 output_count,
    const rund::kernel::u64 window_size, const rund::kernel::u64 stride,
    const rund::kernel::u64 padding, const rund::kernel::u32 element_bytes = 4u,
    const rund::kernel::ComputeDomain domain = rund::kernel::ComputeDomain::U32,
    const rund::node::accel::detail::RangeCount count =
        rund::node::accel::detail::RangeCount::Descriptor) noexcept {
  using namespace rund::node::accel::detail;
  const std::optional<RangeShape> shape = RangeShape::affine(
      Traits(operation, domain), boundary, input_count, output_count,
      window_size, stride, padding, element_bytes, count);
  assert(shape.has_value());
  return *shape;
}

[[nodiscard]] constexpr rund::node::accel::detail::RangeCaps
Gpu(const rund::node::accel::detail::RangeSource variant,
    const std::uint8_t widths = rund::node::accel::detail::kRangeKnownWidthMask,
    const rund::kernel::u32 maximum_threads = 256u,
    const rund::kernel::u32 occupancy = 4u,
    const rund::kernel::u64 shared_limit = 32768u,
    const rund::kernel::u64 maximum_groups =
        std::numeric_limits<rund::kernel::u32>::max(),
    const std::uint8_t support = kAllCandidates,
    const rund::kernel::u64 maximum_storage_elements = 0u,
    const rund::kernel::u64 maximum_storage_bytes =
        std::numeric_limits<rund::kernel::u64>::max()) noexcept {
  using namespace rund::node::accel::detail;
  const rund::kernel::u64 storage_limit =
      maximum_storage_elements != 0u
          ? maximum_storage_elements
          : (variant == RangeSource::Vulkan
                 ? std::numeric_limits<rund::kernel::u32>::max()
                 : std::numeric_limits<rund::kernel::u64>::max());
  const std::optional<RangeCaps> capabilities = RangeCaps::gpu(
      variant, widths, maximum_threads, occupancy, shared_limit, maximum_groups,
      storage_limit, maximum_storage_bytes, support);
  assert(capabilities.has_value());
  return *capabilities;
}

[[nodiscard]] constexpr std::uint8_t
WidthMask(const rund::kernel::u32 width) noexcept {
  using namespace rund::node::accel::detail;
  return width == 64u    ? kRangeWidth64Bit
         : width == 128u ? kRangeWidth128Bit
         : width == 256u ? kRangeWidth256Bit
                         : 0u;
}

[[nodiscard]] constexpr rund::node::accel::detail::RangeSupport
Support(const rund::node::accel::detail::RangePath path) noexcept {
  using namespace rund::node::accel::detail;
  switch (path) {
  case RangePath::Direct:
    return RangeSupport::Direct;
  case RangePath::SharedHalo:
    return RangeSupport::SharedHalo;
  case RangePath::PrefixDifference:
    return RangeSupport::PrefixDifference;
  case RangePath::TiledDifference:
    return RangeSupport::TiledDifference;
  case RangePath::BlockPrefixSuffix:
    return RangeSupport::BlockPrefixSuffix;
  }
  return static_cast<RangeSupport>(0u);
}

// Forced-family source contracts restrict capability support, then invoke the
// production planner. They never construct a candidate, stage graph, cost, or
// identity directly.
[[nodiscard]] constexpr rund::node::accel::detail::RangePlan
PlanSourceVariant(const rund::node::accel::detail::RangeSource backend,
                  const rund::node::accel::detail::RangeOp operation,
                  const rund::kernel::ComputeDomain domain,
                  const rund::kernel::u32 width,
                  const rund::kernel::u32 shared_radius_capacity,
                  const rund::node::accel::detail::RangePath path) noexcept {
  using namespace rund::node::accel::detail;
  const std::optional<RangeTraits> traits = MaybeTraits(operation, domain);
  const rund::kernel::u64 radius =
      path == RangePath::SharedHalo ? shared_radius_capacity : width;
  const rund::kernel::u64 count =
      static_cast<rund::kernel::u64>(width) * 3u + 3u;
  const rund::kernel::u32 element_bytes =
      domain == rund::kernel::ComputeDomain::I64 ||
              domain == rund::kernel::ComputeDomain::U64 ||
              domain == rund::kernel::ComputeDomain::Fixed
          ? 8u
          : 4u;
  if (!traits.has_value()) {
    return RangePlan::rejected("compute_range_aggregate_shape_invalid");
  }
  const RangeShape shape =
      Shape(operation, count, radius, element_bytes, domain);
  const RangeSupport selected = Support(path);
  const std::uint8_t support =
      RangeSupportBit(RangeSupport::Direct) | RangeSupportBit(selected);
  const rund::kernel::u64 shared_bytes =
      path == RangePath::SharedHalo ? static_cast<rund::kernel::u64>(
                                          width + 2u * shared_radius_capacity) *
                                          element_bytes
      : path == RangePath::TiledDifference
          ? static_cast<rund::kernel::u64>(2u * width) * element_bytes
      : path == RangePath::PrefixDifference
          ? static_cast<rund::kernel::u64>(width) * element_bytes
          : 0u;
  const std::optional<RangeCaps> capabilities = RangeCaps::gpu(
      backend, WidthMask(width), width,
      shared_bytes == 0u ? 0u : kRangeSharedReserve,
      shared_bytes == 0u ? 0u : shared_bytes * kRangeSharedReserve,
      std::numeric_limits<rund::kernel::u32>::max(),
      backend == RangeSource::Vulkan
          ? std::numeric_limits<rund::kernel::u32>::max()
          : std::numeric_limits<rund::kernel::u64>::max(),
      support);
  if (!capabilities.has_value()) {
    return RangePlan::rejected("compute_range_aggregate_capabilities_invalid");
  }
  const RangePlan plan = ContractPlanRange(shape, *capabilities);
  return plan.ok() && plan.candidate().disposition() == path
             ? plan
             : RangePlan::rejected(
                   "compute_range_aggregate_candidate_unavailable");
}

[[nodiscard]] constexpr rund::node::accel::detail::RangeExec
RequireExec(const rund::node::accel::detail::RangePlan &plan) noexcept {
  using namespace rund::node::accel::detail;
  const std::optional<RangeExec> execution = RangeExec::from(plan);
  assert(execution.has_value());
  return *execution;
}

[[nodiscard]] bool ModelContract();
[[nodiscard]] bool PlannerContract();
[[nodiscard]] bool ProjectionContract() noexcept;
[[nodiscard]] bool ExecutionContract();
[[nodiscard]] bool MemoryContract();
[[nodiscard]] bool SourceContract();
[[nodiscard]] bool SharedHaloSourceContract();
[[nodiscard]] bool BackendContract();
[[nodiscard]] bool CacheContract();
[[nodiscard]] bool SignedSourcesCarryDomainOrder();
[[nodiscard]] bool SourcesCarryLinearFamilies();
[[nodiscard]] bool MetalRejectedCompileTelemetryIsExact();
[[nodiscard]] bool MetalNamedPipelinePublicationIsTransactional();
[[nodiscard]] bool MetalSourcePublicationIsTransactional();
[[nodiscard]] bool MetalSourceRetryIsExact();

} // namespace node_accel_contract::range

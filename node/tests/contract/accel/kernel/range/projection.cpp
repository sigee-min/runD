#include "local.hpp"

#include "src/accel/metal/range/local.hpp"

#include <limits>

namespace node_accel_contract::range {
namespace {

[[nodiscard]] constexpr bool ProjectionOracle() noexcept {
  using namespace rund::node::accel::detail;
  constexpr std::uint8_t direct = RangeSupportBit(RangeSupport::Direct);
  constexpr std::uint8_t direct_shared =
      direct | RangeSupportBit(RangeSupport::SharedHalo);
  constexpr std::uint8_t direct_prefix =
      direct | RangeSupportBit(RangeSupport::PrefixDifference);
  constexpr std::uint8_t direct_block =
      direct | RangeSupportBit(RangeSupport::BlockPrefixSuffix);
  constexpr auto shared_capabilities = RangeCaps::gpu(
      RangeSource::Vulkan, kRangeWidth128Bit, 128u, 4u,
      (128u + 2u * 7u) * 8u * 4u, std::numeric_limits<rund::kernel::u32>::max(),
      std::numeric_limits<rund::kernel::u32>::max(), direct_shared);
  constexpr auto direct_capabilities =
      RangeCaps::gpu(RangeSource::Metal, kRangeWidth128Bit, 128u, 0u, 0u,
                     std::numeric_limits<rund::kernel::u32>::max(),
                     std::numeric_limits<rund::kernel::u64>::max(), direct);
  constexpr auto prefix_capabilities = RangeCaps::gpu(
      RangeSource::Metal, kRangeWidth64Bit, 64u, 4u, 32768u,
      std::numeric_limits<rund::kernel::u32>::max(),
      std::numeric_limits<rund::kernel::u64>::max(), direct_prefix);
  constexpr auto block_capabilities = RangeCaps::gpu(
      RangeSource::Vulkan, kRangeWidth64Bit, 64u, 0u, 0u,
      std::numeric_limits<rund::kernel::u32>::max(),
      std::numeric_limits<rund::kernel::u32>::max(), direct_block);
  constexpr RangeShape shared_shape =
      Shape(RangeOp::Sum, 129u, 7u, 8u, rund::kernel::ComputeDomain::U64);
  constexpr RangeShape direct_shape =
      Shape(RangeOp::Sum, 65u, 65u, 4u, rund::kernel::ComputeDomain::U32);
  constexpr RangeShape prefix_shape =
      Shape(RangeOp::Sum, 515u, 515u, 4u, rund::kernel::ComputeDomain::U32);
  constexpr RangeShape block_shape =
      Shape(RangeOp::Minimum, 515u, 515u, 4u, rund::kernel::ComputeDomain::I32);
  constexpr RangePlan shared_plan =
      PlanRange(shared_shape, *shared_capabilities);
  constexpr RangePlan direct_plan =
      PlanRange(direct_shape, *direct_capabilities);
  constexpr RangePlan prefix_plan =
      PlanRange(prefix_shape, *prefix_capabilities);
  constexpr RangePlan block_plan = PlanRange(block_shape, *block_capabilities);
  constexpr RangePlan cpu_plan = PlanRange(direct_shape, RangeCaps::cpu());
  constexpr RangePlan unavailable_plan =
      PlanRange(direct_shape, RangeCaps::unavailable());
  constexpr RangePlan metal_direct_range = PlanSourceVariant(
      RangeSource::Metal, RangeOp::Sum, rund::kernel::ComputeDomain::U32, 64u,
      0u, RangePath::Direct);

  constexpr auto shared_candidate = RangeCandidate::shared_halo(128u, 7u);
  constexpr auto direct_candidate = RangeCandidate::direct_gpu(128u);
  constexpr auto prefix_candidate = RangeCandidate::prefix_difference(64u);
  constexpr auto block_candidate = RangeCandidate::block_prefix_suffix(64u);

  return shared_plan.ok() && direct_plan.ok() && prefix_plan.ok() &&
         block_plan.ok() && cpu_plan.ok() && !unavailable_plan.ok() &&
         shared_candidate.has_value() && direct_candidate.has_value() &&
         prefix_candidate.has_value() && block_candidate.has_value() &&
         shared_plan.candidate() == *shared_candidate &&
         direct_plan.candidate() == *direct_candidate &&
         prefix_plan.candidate() == *prefix_candidate &&
         block_plan.candidate() == *block_candidate &&
         !RangeExec::from(cpu_plan).has_value() &&
         !RangeExec::from(unavailable_plan).has_value() &&
         MetalRangeSupports(RequireExec(metal_direct_range),
                            MetalRangeLimits{
                                .maximum_workgroup_width = 64u,
                                .static_shared_bytes = 4u,
                                .shared_memory_limit = 32768u,
                            }) == MetalRangeSupport::Invalid;
}

static_assert(ProjectionOracle());

} // namespace

bool ProjectionContract() noexcept { return ProjectionOracle(); }

} // namespace node_accel_contract::range

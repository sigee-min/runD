#include <accel/api.hpp>
#include <accel/device.hpp>
#include <accel/runtime.hpp>

#include "../range/local.hpp"
#include "local.hpp"
#include "src/accel/context/internal/support.hpp"
#include "src/accel/kernel/preparation.hpp"
#include "src/accel/metal/pipeline/cache.hpp"
#include "src/accel/metal/pipeline/template.hpp"
#include "src/accel/metal/range/local.hpp"
#include "src/accel/range_aggregate/plan.hpp"
#include "src/accel/source/hash.hpp"
#include "src/accel/stencil/shape.hpp"
#include "src/accel/vulkan/kernel/pipeline/template.hpp"
#include "src/accel/vulkan/range/local.hpp"
#include <node/accel/buffer.hpp>
#include <node/accel/pick.hpp>

#include <algorithm>
#include <atomic>
#include <iostream>
#include <limits>
#include <mutex>
#include <optional>
#include <string>
#include <string_view>

namespace node_accel_contract::stencil {

namespace {

[[nodiscard]] constexpr bool PlanProjectionOracle() noexcept {
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
  constexpr RangeShape shared_shape = range::Shape(
      RangeOp::Sum, 129u, 7u, 8u, rund::kernel::ComputeDomain::U64);
  constexpr RangeShape direct_shape = range::Shape(
      RangeOp::Sum, 65u, 65u, 4u, rund::kernel::ComputeDomain::U32);
  constexpr RangeShape prefix_shape = range::Shape(
      RangeOp::Sum, 515u, 515u, 4u, rund::kernel::ComputeDomain::U32);
  constexpr RangeShape block_shape = range::Shape(
      RangeOp::Minimum, 515u, 515u, 4u, rund::kernel::ComputeDomain::I32);
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
  constexpr RangePlan metal_direct_range = range::PlanSourceVariant(
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
         MetalRangeSupports(range::RequireExec(metal_direct_range),
                            MetalRangeLimits{
                                .maximum_workgroup_width = 64u,
                                .static_shared_bytes = 4u,
                                .shared_memory_limit = 32768u,
                            }) == MetalRangeSupport::Invalid;
}

static_assert(PlanProjectionOracle());

} // namespace

bool PlanProjectionContract() noexcept { return PlanProjectionOracle(); }

[[nodiscard]] bool StorageContract() {
  using namespace rund::node::accel::detail;
  constexpr rund::kernel::StencilDesc desc{
      .op = rund::kernel::StencilOp::Sum,
      .element = rund::kernel::StencilElement::U32,
      .boundary = rund::kernel::StencilBoundary::Clamp,
      .element_count = 4u,
      .radius = 1u,
  };
  constexpr rund::kernel::StencilPlan plan = rund::kernel::PlanStencil(desc);
  static_assert(plan.ok);
  rund::kernel::ResidentBufferRef input{
      .id = 41u,
      .bytes = 32u,
      .element_bytes = 4u,
      .stride_bytes = 4u,
      .count = 4u,
      .usage = rund::kernel::kResidentUsageRead,
  };
  rund::kernel::ResidentBufferRef output{
      .id = input.id,
      .bytes = input.bytes,
      .element_bytes = 4u,
      .stride_bytes = 4u,
      .count = 4u,
      .usage = rund::kernel::kResidentUsageWrite,
  };
  const std::shared_ptr<void> owner = std::make_shared<int>(1);
  const RangeBinds bindings{
      .input = &input,
      .input_handle = &owner,
      .output = &output,
      .output_handle = &owner,
  };
  if (StencilShapeOk(desc, plan, bindings)) {
    return false;
  }
  output.offset_bytes = 16u;
  return StencilShapeOk(desc, plan, bindings);
}

} // namespace node_accel_contract::stencil

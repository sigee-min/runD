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
      ContractPlanRange(Shape(RangeOp::Sum, exact - 1u, 1u), vulkan);
  const RangePlan at =
      ContractPlanRange(Shape(RangeOp::Sum, exact, 1u), vulkan);
  const RangePlan above =
      ContractPlanRange(Shape(RangeOp::Sum, exact + 1u, 1u), vulkan);
  const RangePlan input_only_above =
      ContractPlanRange(AffineShape(RangeOp::Sum, RangeBoundary::Clamp,
                                    exact + 1u, 1u, 1u, 1u, 0u),
                        vulkan);
  const RangePlan metal_above =
      ContractPlanRange(Shape(RangeOp::Sum, exact + 1u, 1u),
                        Gpu(RangeSource::Metal, kRangeWidth64Bit, 64u, 0u, 0u,
                            std::numeric_limits<u32>::max(), direct,
                            std::numeric_limits<u64>::max()));
  const RangePlan cpu_above = ContractPlanRange(
      Shape(RangeOp::Sum, exact + 1u, 1u), RangeCaps::cpu_reference());
  const RangePlan identity_wide =
      ContractPlanRange(Shape(RangeOp::Sum, 65u, 1u), vulkan);
  const RangePlan identity_narrow = ContractPlanRange(
      Shape(RangeOp::Sum, 65u, 1u),
      Gpu(RangeSource::Vulkan, kRangeWidth64Bit, 64u, 0u, 0u,
          std::numeric_limits<u32>::max(), direct, exact - 1u));

  const RangeCaps vulkan_block =
      Gpu(RangeSource::Vulkan, kRangeWidth64Bit, 64u, 0u, 0u,
          std::numeric_limits<u32>::max(), direct_block, exact);
  const RangePlan span_at = ContractPlanRange(
      AffineShape(RangeOp::Minimum, RangeBoundary::Clip, 1u, 1u, exact, 1u, 0u),
      vulkan_block);
  const RangePlan span_above =
      ContractPlanRange(AffineShape(RangeOp::Minimum, RangeBoundary::Clip, 1u,
                                    1u, exact + 1u, 1u, 0u),
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
  const RangePlan prefix_below = ContractPlanRange(
      prefix_shape,
      Gpu(RangeSource::Metal, kRangeWidth64Bit, 64u, 4u, 32768u,
          std::numeric_limits<u32>::max(), direct_prefix, 0u, 259u));
  const RangePlan prefix_exact = ContractPlanRange(
      prefix_shape,
      Gpu(RangeSource::Metal, kRangeWidth64Bit, 64u, 4u, 32768u,
          std::numeric_limits<u32>::max(), direct_prefix, 0u, 260u));
  const RangePlan prefix_above = ContractPlanRange(
      prefix_shape,
      Gpu(RangeSource::Metal, kRangeWidth64Bit, 64u, 4u, 32768u,
          std::numeric_limits<u32>::max(), direct_prefix, 0u, 261u));

  const RangeShape block_shape = AffineShape(
      RangeOp::Minimum, RangeBoundary::Clip, 16u, 16u, 33u, 1u, 16u);
  const RangePlan block_below = ContractPlanRange(
      block_shape,
      Gpu(RangeSource::Vulkan, kRangeWidth64Bit, 64u, 0u, 0u,
          std::numeric_limits<u32>::max(), direct_block, 0u, 191u));
  const RangePlan block_exact = ContractPlanRange(
      block_shape,
      Gpu(RangeSource::Vulkan, kRangeWidth64Bit, 64u, 0u, 0u,
          std::numeric_limits<u32>::max(), direct_block, 0u, 192u));
  const RangePlan block_above = ContractPlanRange(
      block_shape,
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

static_assert(StorageIndexCapabilityContract());
static_assert(StorageBindingCapabilityContract());
static_assert(sizeof(RangePlan) <= 320u);

} // namespace

bool MemoryContract() {
  return StorageIndexCapabilityContract() && StorageBindingCapabilityContract();
}

} // namespace node_accel_contract::range

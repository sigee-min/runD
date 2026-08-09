#if defined(__APPLE__) && defined(RUND_NODE_HAVE_METAL_SDK)
#include "../range/local.hpp"
#include "src/accel/metal/compact/local.hpp"
#include "src/accel/metal/gather/local.hpp"
#include "src/accel/metal/histogram/local.hpp"
#include "src/accel/metal/numeric/source.hpp"
#include "src/accel/metal/partition/local.hpp"
#include "src/accel/metal/range/local.hpp"
#include "src/accel/metal/range/pipeline/name.hpp"
#include "src/accel/metal/reduce/local.hpp"
#include "src/accel/metal/scan/source.hpp"
#include "src/accel/metal/scatter/local.hpp"
#include "src/accel/metal/scatter/reduce/model.hpp"
#include "src/accel/metal/segmented/local.hpp"
#include "src/accel/metal/segmented/reduce/model.hpp"
#include "src/accel/metal/sort/source.hpp"
#endif

#include <array>
#include <cstdint>
#include <string>

#include "numeric.hpp"

namespace node_accel_contract {

[[nodiscard]] bool MetalSourceRecipesAreExactAndSemantic() {
#if defined(__APPLE__) && defined(RUND_NODE_HAVE_METAL_SDK)
  using namespace rund::node::accel::detail;
  const auto exact = [](const std::string &source, const std::uint64_t bytes) {
    return !source.empty() && source.size() == bytes;
  };
  std::uint64_t bytes = 0u;
  if (!MetalScanSourceUpperBytes(bytes) || !exact(MetalScanSource(), bytes) ||
      !MetalSegmentedScanSourceUpperBytes(bytes) ||
      !exact(MetalSegmentedScanSource(), bytes) ||
      !MetalSortSourceUpperBytes(256u, bytes) ||
      !exact(MetalSortSource(256u), bytes) ||
      !MetalCompactSourceUpperBytes(bytes) ||
      !exact(MetalCompactSource(), bytes) ||
      !exact(MetalGatherSource(), MetalGatherSourceUpperBytes()) ||
      !exact(MetalHistogramSource(), MetalHistogramSourceUpperBytes()) ||
      !MetalPartitionSourceUpperBytes(bytes) ||
      !exact(MetalPartitionSource(), bytes) ||
      !exact(MetalScatterSource(), MetalScatterSourceUpperBytes()) ||
      !MetalNumericSourceUpperBytes(bytes) ||
      !exact(MetalNumericSource(), bytes)) {
    return false;
  }

  constexpr std::array<rund::kernel::ReduceOp, 4u> reduce_ops{
      rund::kernel::ReduceOp::Sum,
      rund::kernel::ReduceOp::CountNonzero,
      rund::kernel::ReduceOp::Min,
      rund::kernel::ReduceOp::Max,
  };
  constexpr std::array<rund::kernel::ComputeDomain, 2u> domains{
      rund::kernel::ComputeDomain::U32,
      rund::kernel::ComputeDomain::I32,
  };
  for (const rund::kernel::ReduceOp op : reduce_ops) {
    for (const rund::kernel::ComputeDomain domain : domains) {
      if (!MetalSegmentedReduceSourceUpperBytes(op, domain, bytes) ||
          !exact(MetalSegmentedReduceSource(op, domain), bytes) ||
          !MetalReduceSourceUpperBytes(op, 256u, domain, bytes) ||
          !exact(MetalReduceSource(op, 256u, domain), bytes)) {
        return false;
      }
    }
  }
  constexpr std::array<RangeOp, 3u> range_ops{
      RangeOp::Sum,
      RangeOp::Minimum,
      RangeOp::Maximum,
  };
  for (const RangeOp op : range_ops) {
    const std::array<RangePath, 3u> paths =
        op == RangeOp::Sum
            ? std::array{RangePath::Direct, RangePath::SharedHalo,
                         RangePath::PrefixDifference}
            : std::array{RangePath::Direct, RangePath::SharedHalo,
                         RangePath::BlockPrefixSuffix};
    for (const RangePath path : paths) {
      const RangePlan range = node_accel_contract::range::PlanSourceVariant(
          RangeSource::Metal, op, rund::kernel::ComputeDomain::U32, 128u,
          path == RangePath::SharedHalo ? 128u : 0u, path);
      const RangeExec execution =
          node_accel_contract::range::RequireExec(range);
      if (!range.ok() || !MetalRangeSourceUpperBytes(execution, bytes) ||
          !exact(MetalRangeSource(execution), bytes)) {
        return false;
      }
    }
  }

  rund::kernel::ScatterReducePlan scatter{};
  scatter.op = rund::kernel::ScatterReduceOp::Sum;
  scatter.domain = rund::kernel::ComputeDomain::U32;
  scatter.element_bytes = 4u;
  if (!MetalScatterReduceSourceUpperBytes(scatter, bytes) ||
      !exact(MetalScatterReduceSource(scatter), bytes)) {
    return false;
  }
  const std::string modular_sum_key = MetalScatterReduceKey(scatter);
  const std::string modular_sum_source = MetalScatterReduceSource(scatter);
  scatter.domain = rund::kernel::ComputeDomain::I32;
  if (MetalScatterReduceKey(scatter) != modular_sum_key ||
      MetalScatterReduceSource(scatter) != modular_sum_source) {
    return false;
  }
  scatter.domain = rund::kernel::ComputeDomain::Fixed;
  scatter.fixed_format.overflow = rund::kernel::ComputeOverflow::Wrap;
  if (MetalScatterReduceKey(scatter) != modular_sum_key ||
      MetalScatterReduceSource(scatter) != modular_sum_source) {
    return false;
  }
  scatter.fixed_format.overflow = rund::kernel::ComputeOverflow::Saturate;
  if (!MetalScatterReduceSourceUpperBytes(scatter, bytes) ||
      !exact(MetalScatterReduceSource(scatter), bytes) ||
      MetalScatterReduceKey(scatter) == modular_sum_key) {
    return false;
  }

  scatter.op = rund::kernel::ScatterReduceOp::Min;
  scatter.fixed_format.overflow = rund::kernel::ComputeOverflow::Wrap;
  const std::string signed_min_key = MetalScatterReduceKey(scatter);
  scatter.domain = rund::kernel::ComputeDomain::I32;
  if (MetalScatterReduceKey(scatter) != signed_min_key) {
    return false;
  }
  scatter.domain = rund::kernel::ComputeDomain::U32;
  if (MetalScatterReduceKey(scatter) == signed_min_key) {
    return false;
  }

  constexpr RangePlan sum_u32 = node_accel_contract::range::PlanSourceVariant(
      RangeSource::Metal, RangeOp::Sum, rund::kernel::ComputeDomain::U32, 64u,
      64u, RangePath::SharedHalo);
  constexpr RangePlan sum_i32 = node_accel_contract::range::PlanSourceVariant(
      RangeSource::Metal, RangeOp::Sum, rund::kernel::ComputeDomain::I32, 64u,
      64u, RangePath::SharedHalo);
  constexpr RangePlan sum_w128 = node_accel_contract::range::PlanSourceVariant(
      RangeSource::Metal, RangeOp::Sum, rund::kernel::ComputeDomain::U32, 128u,
      128u, RangePath::SharedHalo);
  constexpr RangePlan minimum_i32 =
      node_accel_contract::range::PlanSourceVariant(
          RangeSource::Metal, RangeOp::Minimum,
          rund::kernel::ComputeDomain::I32, 64u, 64u, RangePath::SharedHalo);
  static_assert(sum_u32.ok() && sum_i32.ok() && sum_w128.ok() &&
                minimum_i32.ok());
  const RangeExec sum_u32_execution =
      node_accel_contract::range::RequireExec(sum_u32);
  const RangeExec sum_i32_execution =
      node_accel_contract::range::RequireExec(sum_i32);
  const RangeExec sum_w128_execution =
      node_accel_contract::range::RequireExec(sum_w128);
  const RangeExec minimum_i32_execution =
      node_accel_contract::range::RequireExec(minimum_i32);
  return RangePipelineKey(sum_u32_execution) ==
             "range.aggregate.1.64.64.0.2.0.0.4" &&
         RangePipelineKey(sum_u32_execution) !=
             RangePipelineKey(sum_i32_execution) &&
         RangePipelineKey(sum_u32_execution) !=
             RangePipelineKey(sum_w128_execution) &&
         RangePipelineKey(sum_i32_execution) !=
             RangePipelineKey(minimum_i32_execution);
#else
  return true;
#endif
}

} // namespace node_accel_contract

#include "src/accel/kernel/backend/execute.hpp"
#include "src/accel/kernel/backend/manifest.hpp"
#include "src/accel/kernel/prepared/template/registry.hpp"
#include "src/accel/kernel/publication.hpp"

#include <kernel/program/compute/partition/plan.hpp>
#include <kernel/program/compute/scan/plan.hpp>

#if defined(__APPLE__) && defined(RUND_NODE_HAVE_METAL_SDK)
#include "src/accel/metal/pipeline/guard.hpp"
#include "src/accel/stencil/range.hpp"
#include "src/accel/window/shape.hpp"

#include <kernel/program/compute/stencil/plan.hpp>
#include <kernel/program/compute/window/plan.hpp>

#include "../../range/local.hpp"
#include "src/accel/metal/compact/local.hpp"
#include "src/accel/metal/gather/local.hpp"
#include "src/accel/metal/histogram/local.hpp"
#include "src/accel/metal/kernel/manifest.hpp"
#include "src/accel/metal/kernel/pipeline/source.hpp"
#include "src/accel/metal/numeric/source.hpp"
#include "src/accel/metal/partition/local.hpp"
#include "src/accel/metal/range/local.hpp"
#include "src/accel/metal/reduce/local.hpp"
#include "src/accel/metal/scan/source.hpp"
#include "src/accel/metal/scatter/local.hpp"
#include "src/accel/metal/scatter/reduce/model.hpp"
#include "src/accel/metal/segmented/local.hpp"
#include "src/accel/metal/segmented/reduce/model.hpp"
#include "src/accel/metal/sort/source.hpp"
#include "src/accel/sort/block/metal.hpp"
#endif

#include <cstddef>
#include <cstdint>
#include <limits>
#include <optional>
#include <string>
#include <utility>

#include "../hooks.hpp"
#include "../manifest.hpp"

namespace node_accel_contract {

[[nodiscard]] bool MetalColdManifestIsExact() {
#if defined(__APPLE__) && defined(RUND_NODE_HAVE_METAL_SDK)
  using namespace rund::node::accel::detail;
  rund::kernel::ComputePlan plan{};
  plan.domain = rund::kernel::ComputeDomain::U32;
  const KernelPreparationScope scope{KernelPreparationMode::PipelinePrivate};
  const auto verify =
      [&plan](const KernelExecutionStep &step,
              const std::uint64_t source_builds, const std::uint64_t libraries,
              const std::uint64_t stages, const std::uint64_t binding_slots,
              const std::uint64_t source_bytes) {
        const PreparedBackendManifest manifest =
            BuildMetalBackendManifest(step, plan, nullptr, 1u);
        std::uint64_t dependency_bytes = 0u;
        std::uint64_t dependency_storage_bytes = 0u;
        std::uint64_t dependency_stages = 0u;
        for (std::size_t index = 0u;
             index < manifest.cache_dependency_entry_count; ++index) {
          const PreparedBackendCacheDependency &dependency =
              manifest.source_dependencies[index];
          dependency_bytes += dependency.source_upper_bytes;
          dependency_storage_bytes += dependency.source_storage_upper_bytes;
          dependency_stages += dependency.pipeline_stage_count;
        }
        return manifest.ok && manifest.source_dependencies_complete &&
               manifest.source_build_count == source_builds &&
               manifest.source_library_dependency_count == libraries &&
               manifest.pipeline_stage_count == stages &&
               manifest.capture_binding_slot_upper == binding_slots &&
               manifest.cache_dependency_entry_count == libraries &&
               manifest.cold_cache_source_bytes == source_bytes &&
               dependency_bytes == source_bytes &&
               manifest.cold_cache_source_storage_bytes ==
                   dependency_storage_bytes &&
               dependency_storage_bytes > dependency_bytes &&
               manifest.cold_source_transient_bytes != 0u &&
               dependency_stages == stages &&
               manifest.cold_cache_native_object_count == libraries + stages;
      };
  const auto guarded_size = [](std::string source,
                               const std::uint64_t entry_count) {
    std::uint64_t upper = 0u;
    if (!PipelinePrivateMetalSourceUpperBytes(source.size(), entry_count, true,
                                              upper)) {
      return std::uint64_t{0u};
    }
    source = PipelinePrivateMetalSource(std::move(source), upper);
    return source.size() == upper ? upper : std::uint64_t{0u};
  };

  KernelExecutionStep step{};
  auto &scan = step.operation.set<operation::Scan>();
  scan.desc = rund::kernel::ScanDesc{
      .op = rund::kernel::ScanOp::ExclusiveSum,
      .element = rund::kernel::ScanElement::U32,
      .element_count = 8u,
      .block_size = 256u,
  };
  scan.plan = rund::kernel::PlanScan(scan.desc);
  if (!verify(step, 1u, 1u, 1u, 11u, guarded_size(MetalScanSource(), 7u))) {
    return false;
  }
  scan.desc.element_count = 257u;
  scan.plan = rund::kernel::PlanScan(scan.desc);
  if (!verify(step, 1u, 1u, 3u, 11u, guarded_size(MetalScanSource(), 7u))) {
    return false;
  }
  step.operation.set<operation::SegmentedScan>();
  if (!verify(step, 1u, 1u, 3u, 7u,
              guarded_size(MetalSegmentedScanSource(), 3u))) {
    return false;
  }
  auto &segmented_reduce = step.operation.set<operation::SegmentedReduce>();
  segmented_reduce.plan.op = rund::kernel::ReduceOp::Sum;
  if (!verify(step, 1u, 1u, 4u, 6u,
              guarded_size(MetalSegmentedReduceSource(segmented_reduce.plan.op,
                                                      plan.domain),
                           5u))) {
    return false;
  }
  step.operation.set<operation::Sort>();
  if (!verify(step, 1u, 1u, 5u, 8u,
              guarded_size(MetalSortSource(kMetalSortBlockSize), 7u))) {
    return false;
  }
  step.operation.set<operation::Compact>();
  if (!verify(step, 2u, 2u, 5u, 11u,
              guarded_size(MetalCompactSource(), 4u) +
                  guarded_size(MetalScanSource(), 7u))) {
    return false;
  }
  step.operation.set<operation::Gather>();
  if (!verify(step, 1u, 1u, 2u, 5u, guarded_size(MetalGatherSource(), 3u))) {
    return false;
  }
  auto &histogram = step.operation.set<operation::Histogram>();
  for (const std::uint64_t bins : {256u, 257u}) {
    histogram.plan.bin_count = bins;
    if (!verify(step, 1u, 1u, 2u, 4u,
                guarded_size(MetalHistogramSource(bins), 2u))) {
      return false;
    }
  }
  auto &partition = step.operation.set<operation::Partition>();
  partition.desc = rund::kernel::PartitionDesc{
      .element_count = 8u,
      .flag_bytes = 4u,
      .value_bytes = 4u,
  };
  partition.plan = rund::kernel::PlanPartition(partition.desc);
  if (!verify(step, 3u, 2u, 3u, 11u,
              guarded_size(MetalPartitionSource(), 6u) +
                  guarded_size(MetalScanSource(), 7u))) {
    return false;
  }
  partition.desc.element_count = 1025u;
  partition.plan = rund::kernel::PlanPartition(partition.desc);
  if (!verify(step, 3u, 2u, 5u, 11u,
              guarded_size(MetalPartitionSource(), 6u) +
                  guarded_size(MetalScanSource(), 7u))) {
    return false;
  }
  auto &reduce = step.operation.set<operation::Reduce>();
  reduce.plan.op = rund::kernel::ReduceOp::Sum;
  reduce.plan.block_size = 256u;
  if (!verify(
          step, 1u, 1u, 1u, 6u,
          guarded_size(MetalReduceSource(reduce.plan.op, reduce.plan.block_size,
                                         plan.domain),
                       2u))) {
    return false;
  }
  step.operation.set<operation::Scatter>();
  if (!verify(step, 1u, 1u, 1u, 5u, guarded_size(MetalScatterSource(), 2u))) {
    return false;
  }
  auto &scatter_reduce = step.operation.set<operation::ScatterReduce>();
  scatter_reduce.plan.op = rund::kernel::ScatterReduceOp::Sum;
  scatter_reduce.plan.domain = rund::kernel::ComputeDomain::U32;
  scatter_reduce.plan.element_bytes = 4u;
  if (!verify(
          step, 1u, 1u, 3u, 8u,
          guarded_size(MetalScatterReduceSource(scatter_reduce.plan), 3u))) {
    return false;
  }
  constexpr rund::kernel::StencilDesc stencil_desc{
      .op = rund::kernel::StencilOp::Sum,
      .element = rund::kernel::StencilElement::U32,
      .boundary = rund::kernel::StencilBoundary::Clamp,
      .element_count = 192u,
      .radius = 1u,
  };
  constexpr rund::kernel::StencilPlan stencil_plan =
      rund::kernel::PlanStencil(stencil_desc);
  constexpr std::optional<RangeShape> stencil_shape =
      ProjectStencilRange(stencil_plan, rund::kernel::ComputeDomain::U32);
  constexpr std::optional<RangeCaps> stencil_capabilities =
      RangeCaps::gpu(RangeSource::Metal, kRangeWidth64Bit, 64u, 0u, 0u,
                     std::numeric_limits<rund::kernel::u32>::max(),
                     std::numeric_limits<rund::kernel::u64>::max(),
                     RangeSupportBit(RangeSupport::Direct));
  constexpr RangePlan stencil_range =
      stencil_shape.has_value() && stencil_capabilities.has_value()
          ? node_accel_contract::range::ContractPlanRange(*stencil_shape,
                                                          *stencil_capabilities)
          : RangePlan::rejected(
                "compute_range_aggregate_candidate_unavailable");
  static_assert(stencil_plan.ok && stencil_range.ok() &&
                stencil_range.candidate().disposition() == RangePath::Direct);
  auto &stencil = step.operation.set<operation::Stencil>(
      stencil_desc, stencil_plan, stencil_range);
  if (!verify(
          step, 1u, 1u, 1u, 3u,
          guarded_size(MetalRangeSource(node_accel_contract::range::RequireExec(
                           stencil.range)),
                       4u))) {
    return false;
  }
  constexpr rund::kernel::WindowDesc window_desc{
      .op = rund::kernel::WindowOp::Sum,
      .element = rund::kernel::WindowElement::U32,
      .boundary = rund::kernel::WindowBoundary::Clamp,
      .domain = rund::kernel::ComputeDomain::U32,
      .count_source = rund::kernel::ComputeCountSource::BufferU32,
      .input_count = 4097u,
      .output_count = 4097u,
      .window_size = 8195u,
      .stride = 1u,
      .pad_left = 4097u,
  };
  constexpr rund::kernel::WindowPlan window_plan =
      rund::kernel::PlanWindow(window_desc);
  constexpr std::optional<RangeShape> window_shape =
      WindowRangeShape(window_plan);
  constexpr std::optional<RangeCaps> window_capabilities =
      RangeCaps::gpu(RangeSource::Metal, kRangeWidth64Bit, 64u, 0u, 0u,
                     std::numeric_limits<rund::kernel::u32>::max(),
                     std::numeric_limits<rund::kernel::u64>::max(),
                     std::numeric_limits<rund::kernel::u64>::max(),
                     RangeSupportBit(RangeSupport::Direct) |
                         RangeSupportBit(RangeSupport::PrefixDifference));
  constexpr RangePlan window_range =
      window_shape.has_value() && window_capabilities.has_value()
          ? node_accel_contract::range::ContractPlanRange(*window_shape,
                                                          *window_capabilities)
          : RangePlan::rejected(
                "compute_range_aggregate_candidate_unavailable");
  static_assert(window_plan.ok && window_range.ok() &&
                window_range.candidate().disposition() ==
                    RangePath::PrefixDifference);
  auto &window = step.operation.set<operation::Window>(window_desc, window_plan,
                                                       window_range);
  step.control = rund::kernel::GraphControl{
      .count_source = rund::kernel::GraphControlSource::U32,
      .count_binding = 1u,
      .capacity = window_plan.input_count,
  };
  const RangeExec window_execution = *RangeExec::from(window.range);
  if (!verify(step, 2u, 2u, window.range.stage_count() + 1u, 5u,
              guarded_size(MetalRangeSource(window_execution), 4u) +
                  guarded_size(MetalRangeControlSource(window.range), 1u))) {
    return false;
  }
  step.control = {};
  const std::uint64_t numeric_source_bytes =
      guarded_size(MetalNumericSource(), 10u);
  step.operation.set<operation::Transform>();
  if (!verify(step, 1u, 1u, 1u, 6u, numeric_source_bytes)) {
    return false;
  }
  step.operation.set<operation::Matrix>();
  if (!verify(step, 1u, 1u, 1u, 4u, numeric_source_bytes)) {
    return false;
  }
  step.operation.set<operation::Factor>();
  if (!verify(step, 1u, 1u, 1u, 5u, numeric_source_bytes)) {
    return false;
  }
  step.operation.set<operation::Solve>();
  if (!verify(step, 1u, 1u, 1u, 6u, numeric_source_bytes)) {
    return false;
  }
  step.operation.set<operation::Spectrum>();
  return verify(step, 1u, 1u, 1u, 5u, numeric_source_bytes);
#else
  return true;
#endif
}

} // namespace node_accel_contract

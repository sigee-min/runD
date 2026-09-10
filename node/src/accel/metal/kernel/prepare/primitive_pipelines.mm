#include "primitive_pipelines.hpp"

#include "../../compact/local.hpp"
#include "../../gather/local.hpp"
#include "../../histogram/local.hpp"
#include "../../numeric/state.hpp"
#include "../../partition/local.hpp"
#include "../../range/local.hpp"
#include "../../reduce/local.hpp"
#include "../../scan/kernel/local.hpp"
#include "../../scatter/local.hpp"
#include "../../scatter/reduce/model.hpp"
#include "../../segmented/local.hpp"
#include "../../segmented/reduce/model.hpp"
#include "../../sort/local.hpp"
#include "../../pipeline/template.hpp"

#include <new>

namespace rund::node::accel::detail {

#if defined(__APPLE__) && defined(RUND_NODE_HAVE_METAL_SDK)
namespace {

template <typename Resource>
[[nodiscard]] const Resource *
MetalPrimitiveResource(const std::shared_ptr<void> &resource) noexcept {
  return static_cast<const Resource *>(resource.get());
}

} // namespace

rund::AccelCheck FreezeMetalPrimitivePipelines(
    const rund::kernel::NodeKind kind, const std::shared_ptr<void> &resource,
    std::shared_ptr<const MetalKernelImmutablePipelines> &out) {
  std::shared_ptr<MetalKernelImmutablePipelines> frozen;
  try {
    frozen = std::make_shared<MetalKernelImmutablePipelines>();
  } catch (const std::bad_alloc &) {
    return {false, "compute_pipeline_capacity"};
  }
  switch (kind) {
  case rund::kernel::NodeKind::Scan: {
    const auto *const raw =
        MetalPrimitiveResource<MetalScanEncodeResources>(resource);
    if (raw == nullptr || !raw->prefix_execution.has_value()) {
      break;
    }
    const auto stage_count = static_cast<std::uint32_t>(
        MetalScanPipelineCount(*raw->prefix_execution));
    if (stage_count == 0u) {
      break;
    }
    frozen->stages[0u] = raw->block;
    if (ScanPrefixHasOffset(*raw->prefix_execution)) {
      frozen->stages[1u] = raw->prefix;
      frozen->stages[2u] = raw->offset;
    }
    frozen->count = stage_count;
    break;
  }
  case rund::kernel::NodeKind::SegmentedScan: {
    const auto *const raw =
        MetalPrimitiveResource<MetalSegmentedScanEncodeResources>(resource);
    if (raw == nullptr) {
      break;
    }
    frozen->stages = {raw->block, raw->prefix, raw->offset};
    frozen->count = 3u;
    break;
  }
  case rund::kernel::NodeKind::SegmentedReduce: {
    const auto *const raw =
        MetalPrimitiveResource<MetalSegmentedReduceResources>(resource);
    if (raw == nullptr) {
      break;
    }
    frozen->stages = {raw->pipelines.classify, raw->pipelines.prefix,
                      raw->pipelines.scatter, raw->pipelines.reduce};
    frozen->count = 4u;
    break;
  }
  case rund::kernel::NodeKind::Sort: {
    const auto *const raw =
        MetalPrimitiveResource<MetalSortEncodeResources>(resource);
    if (raw == nullptr) {
      break;
    }
    frozen->stages = {raw->pipelines.dispatch, raw->pipelines.histogram,
                      raw->pipelines.prefix, raw->pipelines.base,
                      raw->pipelines.scatter};
    frozen->count = 5u;
    break;
  }
  case rund::kernel::NodeKind::Compact: {
    const auto *const raw =
        MetalPrimitiveResource<MetalCompactEncodeResources>(resource);
    if (raw == nullptr) {
      break;
    }
    if (raw->block_offset_path) {
      frozen->stages[0u] = raw->pipelines.count_blocks;
      frozen->stages[1u] = raw->pipelines.scatter_blocks;
    } else {
      frozen->stages[0u] = raw->pipelines.scatter;
      frozen->stages[1u] = raw->pipelines.status;
    }
    frozen->stages[2u] = raw->scan_block;
    frozen->stages[3u] = raw->scan_prefix;
    frozen->stages[4u] = raw->scan_offset;
    frozen->count = 5u;
    break;
  }
  case rund::kernel::NodeKind::Gather: {
    const auto *const raw =
        MetalPrimitiveResource<MetalGatherEncodeResources>(resource);
    if (raw == nullptr) {
      break;
    }
    frozen->stages = {raw->control_pipeline, raw->gather_pipeline};
    frozen->count = 2u;
    break;
  }
  case rund::kernel::NodeKind::Histogram: {
    const auto *const raw =
        MetalPrimitiveResource<MetalHistogramEncodeResources>(resource);
    if (raw == nullptr) {
      break;
    }
    frozen->stages = {raw->pipelines.clear, raw->pipelines.count};
    frozen->count = 2u;
    break;
  }
  case rund::kernel::NodeKind::Partition: {
    const auto *const raw =
        MetalPrimitiveResource<MetalPartitionEncodeResources>(resource);
    if (raw == nullptr || !raw->scan_execution.has_value()) {
      break;
    }
    const auto pipeline_count = static_cast<std::uint32_t>(
        MetalPartitionPipelineCount(*raw->scan_execution));
    if (pipeline_count == 0u) {
      break;
    }
    frozen->stages[0u] = raw->pipelines.classify;
    frozen->stages[1u] = raw->pipelines.scatter;
    frozen->stages[2u] = raw->scan_block;
    if (ScanPrefixHasOffset(*raw->scan_execution)) {
      frozen->stages[3u] = raw->scan_prefix;
      frozen->stages[4u] = raw->scan_offset;
    }
    frozen->count = pipeline_count;
    break;
  }
  case rund::kernel::NodeKind::Reduce: {
    const auto *const raw =
        MetalPrimitiveResource<MetalReduceEncodeResources>(resource);
    if (raw == nullptr) {
      break;
    }
    frozen->stages[0u] = raw->pipeline;
    frozen->count = 1u;
    break;
  }
  case rund::kernel::NodeKind::Scatter: {
    const auto *const raw =
        MetalPrimitiveResource<MetalScatterEncodeResources>(resource);
    if (raw == nullptr) {
      break;
    }
    frozen->stages[0u] = raw->pipeline;
    frozen->count = 1u;
    break;
  }
  case rund::kernel::NodeKind::ScatterReduce: {
    const auto *const raw =
        MetalPrimitiveResource<MetalScatterReduceResources>(resource);
    if (raw == nullptr) {
      break;
    }
    frozen->stages = {raw->control_pipeline, raw->init_pipeline,
                      raw->fold_pipeline};
    frozen->count = 3u;
    break;
  }
  case rund::kernel::NodeKind::Stencil:
  case rund::kernel::NodeKind::Window: {
    const auto *const raw =
        MetalPrimitiveResource<MetalRangeResources>(resource);
    if (raw == nullptr || raw->stage_count == 0u ||
        raw->stage_count > frozen->stages.size() ||
        (raw->controlled && raw->control_pipeline == nullptr)) {
      break;
    }
    for (std::size_t index = 0u; index < raw->stage_count; ++index) {
      frozen->stages[index] = raw->pipelines[index];
    }
    frozen->control = raw->control_pipeline;
    frozen->count = raw->stage_count;
    break;
  }
  case rund::kernel::NodeKind::Transform:
  case rund::kernel::NodeKind::Matrix:
  case rund::kernel::NodeKind::Factor:
  case rund::kernel::NodeKind::Solve:
  case rund::kernel::NodeKind::Spectrum: {
    const auto *const raw =
        MetalPrimitiveResource<MetalNumericPrepared>(resource);
    if (raw == nullptr) {
      break;
    }
    frozen->stages[0u] = raw->pipeline;
    frozen->count = 1u;
    break;
  }
  case rund::kernel::NodeKind::Map:
    return {false, "accel_kernel_template_invalid"};
  }
  if (!frozen->ready(frozen->count)) {
    return {false, "accel_metal_pipeline_unavailable"};
  }
  out = std::move(frozen);
  return {true, "ok"};
}
#endif

} // namespace rund::node::accel::detail

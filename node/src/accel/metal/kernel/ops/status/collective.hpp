#pragma once

#include "common.hpp"

#include "../../../../segmented/reduce/metal.hpp"
#include "../../../partition/local.hpp"
#include "../../../reduce/local.hpp"
#include "../../../scan/kernel/local.hpp"
#include "../../../segmented/local.hpp"
#include "../../../sort/local.hpp"

#include <limits>

namespace rund::node::accel::detail {
#if defined(__APPLE__) && defined(RUND_NODE_HAVE_METAL_SDK)

[[nodiscard]] inline bool
DescribeMetalScanPipelineStatus(const std::shared_ptr<void> &resources,
                                MetalPipelineStatusBindings &out) noexcept {
  using namespace metal_pipeline_status;
  out = {};
  const auto *const scan =
      static_cast<const MetalScanEncodeResources *>(resources.get());
  return scan != nullptr &&
         append(
             out,
             binding(scan->status, MetalPipelineStatusEncoding::BitFlags,
                     {reason(rund::compute::Reason::ScanSumOverflow),
                      reason(rund::compute::Reason::BoundedCountInvalid), 0u}));
}

[[nodiscard]] inline bool DescribeMetalScanPipelineTelemetry(
    const std::shared_ptr<void> &resources,
    MetalPipelineTelemetrySource &source) noexcept {
  source = {};
  const auto *const scan =
      static_cast<const MetalScanEncodeResources *>(resources.get());
  if (scan == nullptr) {
    return false;
  }
  if (scan->control.iteration == 0u) {
    return true;
  }
  if (!scan->control.has_count() ||
      scan->logical_count.device_buffer == nullptr) {
    return false;
  }
  source = MetalPipelineTelemetrySource{
      .kind = MetalPipelineTelemetryKind::ControlledCollective,
      .primary_buffer = scan->logical_count.device_buffer.get(),
      .count_buffer = scan->logical_count.device_buffer.get(),
      .control = scan->control,
      .count_offset = scan->logical_count.ref.offset_bytes +
                      scan->control.count_byte_offset,
      .capacity = scan->control.capacity,
  };
  return true;
}

[[nodiscard]] inline bool DescribeMetalSegmentedPipelineStatus(
    const std::shared_ptr<void> &resources,
    MetalPipelineStatusBindings &out) noexcept {
  using namespace metal_pipeline_status;
  out = {};
  const auto *const scan =
      static_cast<const MetalSegmentedScanEncodeResources *>(resources.get());
  return scan != nullptr &&
         append(out,
                binding(
                    scan->status, MetalPipelineStatusEncoding::SegmentedScan,
                    {reason(rund::compute::Reason::SegmentedScanSumOverflow),
                     reason(rund::compute::Reason::SegmentedScanSegmentInvalid),
                     0u}));
}

[[nodiscard]] inline bool
DescribeMetalSortPipelineStatus(const std::shared_ptr<void> &resources,
                                MetalPipelineStatusBindings &out) noexcept {
  using namespace metal_pipeline_status;
  out = {};
  const auto *const sort =
      static_cast<const MetalSortEncodeResources *>(resources.get());
  if (sort == nullptr) {
    return false;
  }
  if (sort->plan.count_source == rund::kernel::ComputeCountSource::Descriptor) {
    return true;
  }
  return append(
      out,
      binding(sort->status, MetalPipelineStatusEncoding::Nonzero,
              {reason(rund::compute::Reason::BoundedCountInvalid), 0u, 0u}));
}

[[nodiscard]] inline bool DescribeMetalSortPipelineTelemetry(
    const std::shared_ptr<void> &resources,
    MetalPipelineTelemetrySource &source) noexcept {
  source = {};
  const auto *const sort =
      static_cast<const MetalSortEncodeResources *>(resources.get());
  if (sort == nullptr) {
    return false;
  }
  if (sort->control.iteration == 0u) {
    return true;
  }
  if (!sort->control.has_count() ||
      sort->logical_count.device_buffer == nullptr ||
      sort->plan.radix_pass_count >
          std::numeric_limits<std::uint32_t>::max() / 2u) {
    return false;
  }
  source = MetalPipelineTelemetrySource{
      .kind = MetalPipelineTelemetryKind::ControlledCollective,
      .primary_buffer = sort->logical_count.device_buffer.get(),
      .count_buffer = sort->logical_count.device_buffer.get(),
      .control = sort->control,
      .count_offset = sort->logical_count.ref.offset_bytes +
                      sort->control.count_byte_offset,
      .capacity = sort->control.capacity,
      .indirect_dispatch_count = sort->plan.radix_pass_count * 2u,
  };
  return true;
}

[[nodiscard]] inline bool DescribeMetalPartitionPipelineStatus(
    const std::shared_ptr<void> &resources,
    MetalPipelineStatusBindings &out) noexcept {
  using namespace metal_pipeline_status;
  out = {};
  const auto *const partition =
      static_cast<const MetalPartitionEncodeResources *>(resources.get());
  return partition != nullptr &&
         append(
             out,
             binding(partition->false_status,
                     MetalPipelineStatusEncoding::BitFlags,
                     {reason(rund::compute::Reason::ScanSumOverflow),
                      reason(rund::compute::Reason::BoundedCountInvalid), 0u}));
}

[[nodiscard]] inline bool
DescribeMetalReducePipelineStatus(const std::shared_ptr<void> &resources,
                                  MetalPipelineStatusBindings &out) noexcept {
  using namespace metal_pipeline_status;
  out = {};
  const auto *const reduce =
      static_cast<const MetalReduceEncodeResources *>(resources.get());
  if (reduce == nullptr) {
    return false;
  }
  const rund::compute::Reason arithmetic =
      reduce->plan.op == rund::kernel::ReduceOp::CountNonzero
          ? rund::compute::Reason::ReduceCountOverflow
          : rund::compute::Reason::ReduceSumOverflow;
  return append(out,
                binding(reduce->status, MetalPipelineStatusEncoding::BitFlags,
                        {reason(arithmetic),
                         reason(rund::compute::Reason::BoundedCountInvalid),
                         reason(rund::compute::Reason::ReduceCountZero)}));
}

#endif
} // namespace rund::node::accel::detail

#pragma once

#include "model.hpp"

#include <rund/compute/reason.hpp>

#include "../../../segmented/reduce/metal.hpp"
#include "../../compact/local.hpp"
#include "../../gather/local.hpp"
#include "../../histogram/local.hpp"
#include "../../partition/local.hpp"
#include "../../reduce/local.hpp"
#include "../../scan/kernel/local.hpp"
#include "../../scatter/local.hpp"
#include "../../segmented/local.hpp"
#include "../../sort/local.hpp"

#include <limits>

namespace rund::node::accel::detail {
#if defined(__APPLE__) && defined(RUND_NODE_HAVE_METAL_SDK)
namespace metal_pipeline_status {

[[nodiscard]] constexpr std::uint32_t
reason(const rund::compute::Reason value) noexcept {
  return static_cast<std::uint32_t>(value);
}

[[nodiscard]] inline MetalPipelineStatusBinding
binding(const MetalRuntimeBuffer &buffer,
        const MetalPipelineStatusEncoding encoding,
        const std::array<std::uint32_t, 4u> reasons,
        const std::uint32_t reset = 0u, const std::uint64_t limit = 0u,
        const std::uint32_t observed = 0u) noexcept {
  return MetalPipelineStatusBinding{
      .buffer = buffer.buffer.get(),
      .offset = buffer.offset,
      .bytes = buffer.bytes,
      .limit = limit,
      .reasons = reasons,
      .reset = reset,
      .observed = observed,
      .encoding = encoding,
  };
}

[[nodiscard]] inline bool append(MetalPipelineStatusBindings &out,
                                 const MetalPipelineStatusBinding value) {
  if (value.buffer == nullptr || value.bytes == 0u ||
      (value.bytes & (sizeof(std::uint32_t) - 1u)) != 0u ||
      value.observed >= value.bytes / sizeof(std::uint32_t) ||
      out.size >= out.values.size()) {
    return false;
  }
  out.values[out.size++] = value;
  return true;
}

} // namespace metal_pipeline_status

[[nodiscard]] inline bool
DescribeMetalMapPipelineStatus(const std::shared_ptr<void> &resources,
                               MetalPipelineStatusBindings &out) noexcept {
  using namespace metal_pipeline_status;
  out = {};
  const auto *const map =
      static_cast<const MetalMapEncodeResources *>(resources.get());
  if (map == nullptr) {
    return false;
  }
  if (!map->controlled()) {
    return true;
  }
  MetalPipelineStatusBinding described =
      binding(map->control_status,
              map->checks.empty() ? MetalPipelineStatusEncoding::Nonzero
                                  : MetalPipelineStatusEncoding::Mapping,
              {reason(rund::compute::Reason::WorksetOverflow),
               reason(rund::compute::Reason::GatherIndexOutOfRange), 0u, 0u});
  if (!map->checks.empty()) {
    described.indirect_dispatch_count = 1u;
  }
  return append(out, described);
}

[[nodiscard]] inline bool DescribeMetalMapPipelineTelemetry(
    const std::shared_ptr<void> &resources,
    MetalPipelineTelemetrySource &source) noexcept {
  source = {};
  const auto *const map =
      static_cast<const MetalMapEncodeResources *>(resources.get());
  if (map == nullptr) {
    return false;
  }
  if (!map->controlled()) {
    return true;
  }
  if (map->control_args.buffer == nullptr || map->windows.empty() ||
      map->windows.size() > std::numeric_limits<std::uint32_t>::max() / 4u) {
    return false;
  }
  source = MetalPipelineTelemetrySource{
      .kind = map->checks.empty() ? MetalPipelineTelemetryKind::ControlledMap
                                  : MetalPipelineTelemetryKind::GatherControl,
      .primary_buffer = map->checks.empty() ? map->control_args.buffer.get()
                                            : map->control_status.buffer.get(),
      .count_buffer = map->control.has_count()
                          ? map->control_count.device_buffer.get()
                          : nullptr,
      .predicate_buffer = map->control.has_predicate()
                              ? map->control_predicate.device_buffer.get()
                              : nullptr,
      .control = map->control,
      .count_offset =
          map->control_count.ref.offset_bytes + map->control.count_byte_offset,
      .predicate_offset = map->control_predicate.ref.offset_bytes +
                          map->control.predicate_byte_offset,
      .capacity = map->control.capacity,
      .work_item_count =
          map->windows.back().begin_sequence + map->windows.back().tile_count,
      .primary_word_count = map->checks.empty() ? static_cast<std::uint32_t>(
                                                      map->windows.size() * 4u)
                                                : 2u,
      .indirect_dispatch_count =
          static_cast<std::uint32_t>(1u + (!map->checks.empty() ? 1u : 0u)),
  };
  return (!map->control.has_count() || source.count_buffer != nullptr) &&
         (!map->control.has_predicate() || source.predicate_buffer != nullptr);
}

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

[[nodiscard]] inline bool
DescribeMetalCompactPipelineStatus(const std::shared_ptr<void> &resources,
                                   MetalPipelineStatusBindings &out) noexcept {
  using namespace metal_pipeline_status;
  out = {};
  const auto *const compact =
      static_cast<const MetalCompactEncodeResources *>(resources.get());
  if (compact == nullptr ||
      !append(
          out,
          binding(compact->scan_status, MetalPipelineStatusEncoding::BitFlags,
                  {reason(rund::compute::Reason::ScanSumOverflow),
                   reason(rund::compute::Reason::BoundedCountInvalid), 0u}))) {
    return false;
  }
  return compact->plan.status_bytes == 0u ||
         append(out,
                binding(
                    compact->status, MetalPipelineStatusEncoding::Limit,
                    {reason(rund::compute::Reason::CompactCapacityInsufficient),
                     0u, 0u},
                    0u, compact->plan.output_capacity));
}

[[nodiscard]] inline bool
DescribeMetalGatherPipelineStatus(const std::shared_ptr<void> &resources,
                                  MetalPipelineStatusBindings &out) noexcept {
  using namespace metal_pipeline_status;
  out = {};
  const auto *const gather =
      static_cast<const MetalGatherEncodeResources *>(resources.get());
  if (gather == nullptr) {
    return false;
  }
  MetalPipelineStatusBinding described =
      binding(gather->status, MetalPipelineStatusEncoding::Mapping,
              {reason(rund::compute::Reason::BoundedCountInvalid),
               reason(rund::compute::Reason::GatherIndexOutOfRange), 0u, 0u},
              0u, gather->plan.element_count);
  described.indirect_dispatch_count = 1u;
  described.telemetry = MetalPipelineStatusTelemetry::Gather;
  return append(out, described);
}

[[nodiscard]] inline bool DescribeMetalHistogramPipelineStatus(
    const std::shared_ptr<void> &resources,
    MetalPipelineStatusBindings &out) noexcept {
  using namespace metal_pipeline_status;
  out = {};
  const auto *const histogram =
      static_cast<const MetalHistogramEncodeResources *>(resources.get());
  return histogram != nullptr &&
         append(out,
                binding(histogram->status,
                        MetalPipelineStatusEncoding::Sentinel,
                        {reason(rund::compute::Reason::HistogramBinInvalid), 0u,
                         0u},
                        std::numeric_limits<std::uint32_t>::max()));
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

[[nodiscard]] inline bool
DescribeMetalScatterPipelineStatus(const std::shared_ptr<void> &resources,
                                   MetalPipelineStatusBindings &out) noexcept {
  using namespace metal_pipeline_status;
  out = {};
  const auto *const scatter =
      static_cast<const MetalScatterEncodeResources *>(resources.get());
  return scatter != nullptr &&
         append(out,
                binding(scatter->status, MetalPipelineStatusEncoding::Scatter,
                        {reason(rund::compute::Reason::ScatterIndexOutOfRange),
                         reason(rund::compute::Reason::ScatterDuplicateIndex),
                         reason(rund::compute::Reason::ScatterInvalid)},
                        std::numeric_limits<std::uint32_t>::max()));
}

#endif
} // namespace rund::node::accel::detail

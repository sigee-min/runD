#pragma once

#include "common.hpp"

#include "../../../runtime/map/resources.hpp"
#include "../../../range/local.hpp"

#include <limits>

namespace rund::node::accel::detail {
#if defined(__APPLE__) && defined(RUND_NODE_HAVE_METAL_SDK)

[[nodiscard]] inline bool
DescribeMetalMapPipelineStatus(const std::shared_ptr<void> &resources,
                               MetalPipelineStatusBindings &out) noexcept {
  using namespace metal_pipeline_status;
  out = {};
  const auto *const map =
      static_cast<const MetalMapEncodeResources *>(resources.get());
  if (map == nullptr || map->prepared == nullptr) {
    return false;
  }
  if (!map->controlled()) {
    return true;
  }
  MetalPipelineStatusBinding described = binding(
      map->control_status,
      map->prepared->checks.empty() ? MetalPipelineStatusEncoding::Nonzero
                                    : MetalPipelineStatusEncoding::Mapping,
      {reason(rund::compute::Reason::WorksetOverflow),
       reason(rund::compute::Reason::GatherIndexOutOfRange), 0u, 0u});
  if (!map->prepared->checks.empty()) {
    described.indirect_dispatch_count = 1u;
  }
  return append(out, described);
}

[[nodiscard]] inline bool
DescribeMetalRangePipelineStatus(const std::shared_ptr<void> &resources,
                                 MetalPipelineStatusBindings &out) noexcept {
  using namespace metal_pipeline_status;
  out = {};
  const auto *const range =
      static_cast<const MetalRangeResources *>(resources.get());
  if (range == nullptr) {
    return false;
  }
  return !range->controlled ||
         append(out, binding(range->control_status,
                             MetalPipelineStatusEncoding::Nonzero,
                             {reason(rund::compute::Reason::WorksetOverflow),
                              0u, 0u, 0u}));
}

[[nodiscard]] inline bool DescribeMetalRangePipelineTelemetry(
    const std::shared_ptr<void> &resources,
    MetalPipelineTelemetrySource &source) noexcept {
  source = {};
  const auto *const range =
      static_cast<const MetalRangeResources *>(resources.get());
  if (range == nullptr) {
    return false;
  }
  if (!range->controlled) {
    return true;
  }
  if (range->control_indirect.buffer == nullptr ||
      range->control_count.device_buffer == nullptr ||
      range->stage_count > std::numeric_limits<std::uint32_t>::max() / 8u) {
    return false;
  }
  source = MetalPipelineTelemetrySource{
      .kind = MetalPipelineTelemetryKind::ControlledRange,
      .primary_buffer = range->control_indirect.buffer.get(),
      .count_buffer = range->control_count.device_buffer.get(),
      .control = range->control,
      .count_offset = range->control_count.ref.offset_bytes +
                      range->control.count_byte_offset,
      .capacity = range->control.capacity,
      .primary_word_count = range->stage_count * 8u,
      .indirect_dispatch_count = range->indirect ? range->stage_count : 0u,
  };
  return true;
}

[[nodiscard]] inline bool DescribeMetalMapPipelineTelemetry(
    const std::shared_ptr<void> &resources,
    MetalPipelineTelemetrySource &source) noexcept {
  source = {};
  const auto *const map =
      static_cast<const MetalMapEncodeResources *>(resources.get());
  if (map == nullptr || map->prepared == nullptr) {
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
      .kind = map->prepared->checks.empty()
                  ? MetalPipelineTelemetryKind::ControlledMap
                  : MetalPipelineTelemetryKind::GatherControl,
      .primary_buffer = map->prepared->checks.empty()
                            ? map->control_args.buffer.get()
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
      .primary_word_count =
          map->prepared->checks.empty()
              ? static_cast<std::uint32_t>(map->windows.size() * 4u)
              : 2u,
      .indirect_dispatch_count = static_cast<std::uint32_t>(
          1u + (!map->prepared->checks.empty() ? 1u : 0u)),
  };
  return (!map->control.has_count() || source.count_buffer != nullptr) &&
         (!map->control.has_predicate() || source.predicate_buffer != nullptr);
}

#endif
} // namespace rund::node::accel::detail

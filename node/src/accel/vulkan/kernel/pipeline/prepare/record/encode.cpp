#include "../record.hpp"
#include "encode/local.hpp"

#include "../../../../command/capture.hpp"
#include "../../../../command/timestamp.hpp"
#include "../../../../command.hpp"
#include "../../../../command/resources.hpp"
#include "../../../../map/api.hpp"
#include "../../../../map/local.hpp"
#include "../../../lease.hpp"
#include "../../telemetry.hpp"

#include <limits>

namespace rund::node::accel::detail {

#if defined(RUND_NODE_HAVE_VULKAN_SDK)
rund::AccelCheck
RecordVulkanPipeline(VulkanPipeline &pipeline, VulkanCommand &command,
                     const CommandKind kind, const bool replay,
                     PreparedPipelineFailureContext *const failure,
                     VulkanTimestampCapture *const timestamps,
                     const VulkanPipelineRecordSlice *const slice) noexcept {
  if (pipeline.adapter == nullptr || pipeline.record == nullptr) {
    return rund::AccelCheck{false, "accel_kernel_pipeline_invalid"};
  }
  VulkanPipelineRecordRecipe &recipe = *pipeline.record;
  const std::size_t first = slice == nullptr ? 0u : slice->first;
  const std::size_t count =
      slice == nullptr ? recipe.entries.size() : slice->count;
  const bool open = slice == nullptr || slice->open;
  const bool close = slice == nullptr || slice->close;
  if (first > recipe.entries.size() || count > recipe.entries.size() - first ||
      (open && first != 0u) ||
      (close && first + count != recipe.entries.size()) ||
      (!open && !close && count != 1u) || (replay && slice != nullptr) ||
      (failure != nullptr && slice != nullptr) ||
      (timestamps != nullptr && slice != nullptr) ||
      (slice != nullptr && slice->capture != nullptr &&
       (open || close || count != 1u))) {
    return rund::AccelCheck{false, "accel_kernel_pipeline_invalid"};
  }
  const std::size_t end = first + count;
  const rund::AccelCheck begun =
      BeginCommand(pipeline.adapter->device, command, kind);
  if (!begun.ok) {
    return begun;
  }
  VulkanLeaseScope recording_leases{*pipeline.adapter,
                                    pipeline.window.descriptor_leases};
  const VkCommandBuffer recording = command.buffer;
  if (timestamps != nullptr) {
    vkCmdResetQueryPool(recording, timestamps->queries, 0u,
                        timestamps->capacity);
  }
  if (open && !OpenVulkanPipelineControl(recording, pipeline.control)) {
    return rund::AccelCheck{false, "accel_vulkan_command_unavailable"};
  }
  if (open && pipeline.profile != nullptr) {
    if (!ResetVulkanPipelineProfile(recording, pipeline.control)) {
      return rund::AccelCheck{false, "accel_vulkan_command_unavailable"};
    }
  }
  if (open && !pipeline.window.routes.empty() &&
      !EncodeVulkanWindowStart(recording, pipeline.window)) {
    return rund::AccelCheck{false, "accel_vulkan_command_unavailable"};
  }
  if (open && recipe.recurrence) {
    const rund::AccelCheck encoded =
        vulkan_record_detail::Trace(timestamps, [&] {
          return EncodeVulkanMap(*pipeline.adapter, pipeline.recurrence,
                                 reinterpret_cast<void *>(recording));
        });
    if (!encoded.ok) {
      return encoded;
    }
  }
  VulkanDispatchCapture replay_capture = pipeline.window.capture;
  if (replay) {
    replay_capture.cursor = 0u;
    replay_capture.indirect_count = 0u;
    replay_capture.failed = false;
    replay_capture.replay = true;
  }
  VulkanDispatchCapture &capture =
      replay ? replay_capture : pipeline.window.capture;
  if (!open) {
    EncodeVulkanComputeToComputeBarrier(recording);
  }
  bool scratch_seen = false;
  const vulkan_record_detail::Encoding context{
      pipeline, recipe, recording, capture, failure, timestamps, slice};
  for (std::size_t index = first; !recipe.recurrence && index < end; ++index) {
    const rund::AccelCheck encoded = vulkan_record_detail::EncodeEntry(
        context, index, index == first, scratch_seen);
    if (!encoded.ok)
      return encoded;
  }
  if (failure != nullptr) {
    failure->stage(PreparedPipelineFailureStage::BackendFinalization);
  }
  if ((slice == nullptr || close) && capture.failed) {
    return rund::AccelCheck{false, "compute_pipeline_capacity"};
  }
  if ((slice == nullptr || close) &&
      (capture.cursor != recipe.window_dispatches ||
       capture.indirect_count != recipe.window_gate_count ||
       pipeline.window.gates.size() != recipe.window_gate_count)) {
    return rund::AccelCheck{false, "compute_dispatch_count_mismatch"};
  }
  if (close && !replay && !FreezeVulkanWindow(pipeline.window)) {
    return rund::AccelCheck{false, "accel_vulkan_memory_unavailable"};
  }
  if (close &&
      (!FinishVulkanPipelineControl(recording, pipeline.control,
                                    recipe.status) ||
       !vulkan_record_detail::Trace(timestamps,
                                    [&] {
                                      return EncodeVulkanPipelinePublish(
                                          recording, pipeline.publish);
                                    }) ||
       !PublishVulkanPipelineControl(recording, pipeline.control))) {
    return rund::AccelCheck{false, "accel_vulkan_command_unavailable"};
  }
  return EndCommand(command);
}

#endif

} // namespace rund::node::accel::detail

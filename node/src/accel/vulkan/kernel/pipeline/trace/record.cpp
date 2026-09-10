#include "../trace.hpp"

#include "../prepare/record.hpp"

#include "../../../command/timestamp.hpp"
#include "../../../command/resources.hpp"
#include "../../../runtime/timestamp.hpp"

#include <limits>

namespace rund::node::accel::detail {

#if defined(RUND_NODE_HAVE_VULKAN_SDK)
namespace {

void DestroyCandidate(VulkanPipeline &pipeline,
                      VulkanPipelineDispatchTrace &trace) noexcept {
  if (trace.queries != VK_NULL_HANDLE) {
    vkDestroyQueryPool(pipeline.adapter->device, trace.queries, nullptr);
  }
  DestroyCommand(pipeline.adapter->device, trace.command);
  trace = {};
}

} // namespace

rund::AccelCheck
EnsureVulkanPipelineDispatchTrace(VulkanPipeline &pipeline) noexcept {
  if (pipeline.trace.ready) {
    return rund::AccelCheck{true, "ok"};
  }
  if (pipeline.adapter != nullptr &&
      pipeline.adapter->fault_trace_unavailable_once.exchange(
          false, std::memory_order_relaxed)) {
    return rund::AccelCheck{false, "compute_telemetry_trace_unavailable"};
  }
  if (pipeline.adapter == nullptr || pipeline.record == nullptr ||
      !VulkanTimestampAvailable(*pipeline.adapter) ||
      pipeline.dispatch_count == 0u) {
    return rund::AccelCheck{false, "compute_telemetry_trace_unavailable"};
  }
  if (pipeline.dispatch_count >
      std::numeric_limits<std::uint32_t>::max() / 2u) {
    return rund::AccelCheck{false, "compute_pipeline_capacity"};
  }
  VulkanPipelineDispatchTrace candidate{};
  candidate.query_count =
      static_cast<std::uint32_t>(2u * pipeline.dispatch_count);
  VkQueryPoolCreateInfo query{};
  query.sType = VK_STRUCTURE_TYPE_QUERY_POOL_CREATE_INFO;
  query.queryType = VK_QUERY_TYPE_TIMESTAMP;
  query.queryCount = candidate.query_count;
  const rund::AccelCheck created = CreateCommand(
      pipeline.adapter->device, pipeline.adapter->compute_queue_family,
      candidate.command, CommandKind::ImmutableSecondary);
  if (!created.ok ||
      vkCreateQueryPool(pipeline.adapter->device, &query, nullptr,
                        &candidate.queries) != VK_SUCCESS ||
      candidate.queries == VK_NULL_HANDLE) {
    DestroyCandidate(pipeline, candidate);
    return rund::AccelCheck{false, "compute_device_capacity"};
  }
  try {
    candidate.values.resize(candidate.query_count);
  } catch (...) {
    DestroyCandidate(pipeline, candidate);
    return rund::AccelCheck{false, "compute_pipeline_capacity"};
  }
  VulkanTimestampCapture capture{.queries = candidate.queries,
                                 .capacity = candidate.query_count};
  const rund::AccelCheck recorded = RecordVulkanPipeline(
      pipeline, candidate.command, CommandKind::ImmutableSecondary, true,
      nullptr, &capture);
  if (!recorded.ok || capture.failed ||
      capture.cursor != candidate.query_count) {
    DestroyCandidate(pipeline, candidate);
    return recorded.ok
               ? rund::AccelCheck{false, "compute_telemetry_trace_unavailable"}
               : recorded;
  }
  candidate.ready = true;
  pipeline.trace = std::move(candidate);
  if (pipeline.memory_meter != nullptr) {
    const std::uint64_t bytes =
        static_cast<std::uint64_t>(pipeline.trace.query_count) *
        sizeof(std::uint64_t);
    pipeline.memory_meter->add(PreparedPipelineMemory{
        .host = {.current = bytes,
                 .peak = bytes,
                 .cumulative = bytes,
                 .budget = bytes},
        .device = {.current = bytes,
                   .peak = bytes,
                   .cumulative = bytes,
                   .budget = bytes},
    });
  }
  return rund::AccelCheck{true, "ok"};
}

rund::AccelCheck
EncodeVulkanPipelineDispatchTrace(VulkanPipeline &pipeline) noexcept {
  if (!pipeline.trace.ready ||
      pipeline.trace.command.buffer == VK_NULL_HANDLE ||
      pipeline.adapter == nullptr ||
      pipeline.adapter->command_buffer == VK_NULL_HANDLE) {
    return rund::AccelCheck{false, "compute_telemetry_trace_unavailable"};
  }
  vkCmdExecuteCommands(pipeline.adapter->command_buffer, 1u,
                       &pipeline.trace.command.buffer);
  return rund::AccelCheck{true, "ok"};
}

#endif

} // namespace rund::node::accel::detail

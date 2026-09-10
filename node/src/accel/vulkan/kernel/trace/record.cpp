#include "../trace.hpp"

#include "../../command/resources.hpp"
#include "../../command/timestamp.hpp"
#include "../../runtime/timestamp.hpp"
#include "../local.hpp"

#include <kernel/core/checked.hpp>

#include <limits>

namespace rund::node::accel::detail {

#if defined(RUND_NODE_HAVE_VULKAN_SDK)

void DestroyVulkanDispatchTrace(VulkanAdapter &adapter,
                                VulkanDispatchTrace &trace) noexcept {
  if (trace.queries != VK_NULL_HANDLE) {
    vkDestroyQueryPool(adapter.device, trace.queries, nullptr);
  }
  DestroyCommand(adapter.device, trace.command);
  trace = {};
}

rund::AccelCheck
EnsureVulkanDispatchTrace(VulkanAdapter &adapter,
                          VulkanKernelResources &resources,
                          PreparedMemoryMeter *const memory) noexcept {
  VulkanDispatchTrace &trace = resources.trace;
  if (trace.ready) {
    return rund::AccelCheck{true, "ok"};
  }
  if (adapter.fault_trace_unavailable_once.exchange(
          false, std::memory_order_relaxed)) {
    return rund::AccelCheck{false, "compute_telemetry_trace_unavailable"};
  }
  if (!VulkanTimestampAvailable(adapter) || resources.dispatch_count == 0u) {
    return rund::AccelCheck{false, "compute_telemetry_trace_unavailable"};
  }
  if (resources.dispatch_count >
      std::numeric_limits<std::uint32_t>::max() / 2u) {
    return rund::AccelCheck{false, "compute_pipeline_capacity"};
  }
  const std::uint32_t query_count =
      static_cast<std::uint32_t>(2u * resources.dispatch_count);
  VulkanDispatchTrace candidate{};
  VkQueryPoolCreateInfo query{};
  query.sType = VK_STRUCTURE_TYPE_QUERY_POOL_CREATE_INFO;
  query.queryType = VK_QUERY_TYPE_TIMESTAMP;
  query.queryCount = query_count;
  const rund::AccelCheck command =
      CreateCommand(adapter.device, adapter.compute_queue_family,
                    candidate.command, CommandKind::ImmutableSecondary);
  if (!command.ok ||
      vkCreateQueryPool(adapter.device, &query, nullptr, &candidate.queries) !=
          VK_SUCCESS ||
      candidate.queries == VK_NULL_HANDLE) {
    DestroyVulkanDispatchTrace(adapter, candidate);
    return rund::AccelCheck{false, "compute_device_capacity"};
  }
  try {
    candidate.values.resize(query_count);
  } catch (...) {
    DestroyVulkanDispatchTrace(adapter, candidate);
    return rund::AccelCheck{false, "compute_pipeline_capacity"};
  }
  const rund::AccelCheck begun = BeginCommand(adapter.device, candidate.command,
                                              CommandKind::ImmutableSecondary);
  if (!begun.ok) {
    DestroyVulkanDispatchTrace(adapter, candidate);
    return begun;
  }
  vkCmdResetQueryPool(candidate.command.buffer, candidate.queries, 0u,
                      query_count);
  VulkanTimestampCapture capture{.queries = candidate.queries,
                                 .capacity = query_count};
  rund::AccelCheck encoded{};
  {
    VulkanTimestampScope scope{capture};
    encoded =
        EncodeVulkanKernelSteps(adapter, resources, candidate.command.buffer);
  }
  if (!encoded.ok || capture.failed || capture.cursor != query_count) {
    DestroyVulkanDispatchTrace(adapter, candidate);
    return encoded.ok
               ? rund::AccelCheck{false, "compute_telemetry_trace_unavailable"}
               : encoded;
  }
  const rund::AccelCheck ended = EndCommand(candidate.command);
  if (!ended.ok) {
    DestroyVulkanDispatchTrace(adapter, candidate);
    return ended;
  }
  candidate.query_count = query_count;
  candidate.ready = true;
  trace = std::move(candidate);
  if (memory != nullptr) {
    const std::uint64_t host = static_cast<std::uint64_t>(
        trace.values.capacity() * sizeof(std::uint64_t));
    memory->add(PreparedMemory{
        .current = host, .peak = host, .cumulative = host, .budget = host});
  }
  return rund::AccelCheck{true, "ok"};
}

rund::AccelCheck
EncodeVulkanDispatchTrace(VulkanAdapter &adapter,
                          VulkanKernelResources &resources) noexcept {
  if (!resources.trace.ready ||
      resources.trace.command.buffer == VK_NULL_HANDLE ||
      adapter.command_buffer == VK_NULL_HANDLE) {
    return rund::AccelCheck{false, "compute_telemetry_trace_unavailable"};
  }
  vkCmdExecuteCommands(adapter.command_buffer, 1u,
                       &resources.trace.command.buffer);
  return rund::AccelCheck{true, "ok"};
}

#endif

} // namespace rund::node::accel::detail

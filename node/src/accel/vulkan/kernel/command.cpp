#include <accel/check.hpp>

#include "../command/resources.hpp"
#include "local.hpp"
#include "ops/table.hpp"

#include <algorithm>

namespace rund::node::accel::detail {

#if defined(RUND_NODE_HAVE_VULKAN_SDK)
rund::AccelCheck EncodeVulkanStep(VulkanAdapter &adapter,
                                  VulkanKernelEntry &entry,
                                  const VkCommandBuffer command) {
  if (entry.ops.encode == nullptr || command == VK_NULL_HANDLE) {
    return rund::AccelCheck{false, "accel_kernel_primitive_unsupported"};
  }
  if (entry.barrier_before && entry.resets.empty()) {
    EncodeVulkanComputeToComputeBarrier(command);
  }
  const rund::AccelCheck gathered = EncodeVulkanViewInputs(entry.view, command);
  if (!gathered.ok) {
    return gathered;
  }
  const rund::AccelCheck encoded = entry.ops.encode(
      adapter, entry.resource, reinterpret_cast<void *>(command));
  if (!encoded.ok) {
    return encoded;
  }
  return EncodeVulkanViewOutputs(entry.view, command);
}

rund::AccelCheck EncodeVulkanResets(VulkanKernelResources &resources,
                                    const std::size_t step,
                                    const VkCommandBuffer command) {
  if (command == VK_NULL_HANDLE) {
    return rund::AccelCheck{false, "accel_vulkan_command_unavailable"};
  }
  VulkanKernelEntry *const entry =
      step < resources.size() ? resources.entry(step) : nullptr;
  if (entry == nullptr || entry->resets.begin > resources.resets.size() ||
      entry->resets.count > resources.resets.size() - entry->resets.begin) {
    return rund::AccelCheck{false, "accel_kernel_reset_invalid"};
  }
  if (entry->resets.empty()) {
    return rund::AccelCheck{true, "ok"};
  }
  const std::size_t end = entry->resets.begin + entry->resets.count;
  bool transfer = false;
  bool compute = false;
  const bool captured = CapturesVulkanDispatch(command);
  for (std::size_t index = entry->resets.begin; index < end; ++index) {
    const VulkanReset &clear = resources.resets[index];
    transfer = transfer || (clear.range.dense() && !captured);
    compute = compute || !clear.range.dense() || captured;
  }
  VkMemoryBarrier writable{};
  writable.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER;
  writable.srcAccessMask =
      VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT;
  writable.dstAccessMask = (transfer ? VK_ACCESS_TRANSFER_WRITE_BIT : 0u) |
                           (compute ? VK_ACCESS_SHADER_WRITE_BIT : 0u);
  vkCmdPipelineBarrier(
      command, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
      (transfer ? VK_PIPELINE_STAGE_TRANSFER_BIT : 0u) |
          (compute ? VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT : 0u),
      0u, 1u, &writable, 0u, nullptr, 0u, nullptr);
  for (std::size_t index = entry->resets.begin; index < end; ++index) {
    const VulkanReset &clear = resources.resets[index];
    if (clear.resident.device_buffer == nullptr ||
        clear.resident.device_buffer->buffer == VK_NULL_HANDLE) {
      return rund::AccelCheck{false, "accel_kernel_reset_invalid"};
    }
    if (clear.range.dense() && !captured) {
      vkCmdFillBuffer(command, clear.resident.device_buffer->buffer,
                      clear.range.offset(), reset::Payload(clear.range), 0u);
      continue;
    }
    if (resources.adapter == nullptr ||
        resources.adapter->max_dispatch_groups == 0u ||
        resources.reset_pipeline == nullptr ||
        clear.descriptor == VK_NULL_HANDLE) {
      return rund::AccelCheck{false, "accel_kernel_reset_invalid"};
    }
    const std::uint64_t window =
        static_cast<std::uint64_t>(resources.adapter->max_dispatch_groups) *
        256u;
    BindVulkanPipeline(command, VK_PIPELINE_BIND_POINT_COMPUTE,
                       resources.reset_pipeline->pipeline);
    BindVulkanDescriptors(command, VK_PIPELINE_BIND_POINT_COMPUTE,
                          resources.reset_pipeline->pipeline_layout, 0u, 1u,
                          &clear.descriptor, 0u, nullptr);
    for (std::uint64_t base = 0u; base < clear.range.count(); base += window) {
      const std::uint64_t count = std::min(window, clear.range.count() - base);
      const reset::Params params =
          reset::Bind(clear.range, base, clear.binding_offset);
      PushVulkanConstants(command, resources.reset_pipeline->pipeline_layout,
                          VK_SHADER_STAGE_COMPUTE_BIT, 0u, sizeof(params),
                          &params);
      const std::uint64_t groups = (count + 255u) / 256u;
      DispatchVulkan(command, static_cast<std::uint32_t>(groups), 1u, 1u);
    }
  }
  VkMemoryBarrier reset_barrier{};
  reset_barrier.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER;
  reset_barrier.srcAccessMask = (transfer ? VK_ACCESS_TRANSFER_WRITE_BIT : 0u) |
                                (compute ? VK_ACCESS_SHADER_WRITE_BIT : 0u);
  reset_barrier.dstAccessMask =
      VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT;
  vkCmdPipelineBarrier(
      command,
      (transfer ? VK_PIPELINE_STAGE_TRANSFER_BIT : 0u) |
          (compute ? VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT : 0u),
      VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0u, 1u, &reset_barrier, 0u, nullptr,
      0u, nullptr);
  return rund::AccelCheck{true, "ok"};
}

rund::AccelCheck EncodeVulkanKernelSteps(VulkanAdapter &adapter,
                                         VulkanKernelResources &resources,
                                         const VkCommandBuffer command) {
  for (std::size_t index = 0u; index < resources.size(); ++index) {
    VulkanKernelEntry *const entry = resources.entry(index);
    if (entry == nullptr) {
      return rund::AccelCheck{false, "accel_kernel_run_invalid"};
    }
    const rund::AccelCheck reset =
        EncodeVulkanResets(resources, index, command);
    if (!reset.ok) {
      return reset;
    }
    const rund::AccelCheck encoded = EncodeVulkanStep(adapter, *entry, command);
    if (!encoded.ok) {
      return encoded;
    }
  }
  return rund::AccelCheck{true, "ok"};
}

void DestroyVulkanKernelCommand(VulkanAdapter &adapter,
                                VulkanKernelResources &resources) noexcept {
  DestroyCommand(adapter.device, resources.command);
}

rund::AccelCheck RecordVulkanKernel(VulkanAdapter &adapter,
                                    VulkanKernelResources &resources) {
  if (resources.size() == 0u || resources.command.pool != VK_NULL_HANDLE ||
      resources.command.buffer != VK_NULL_HANDLE ||
      resources.command.fence != VK_NULL_HANDLE) {
    return rund::AccelCheck{false, "accel_vulkan_command_unavailable"};
  }

  const rund::AccelCheck created =
      CreateCommand(adapter.device, adapter.compute_queue_family,
                    resources.command, CommandKind::ImmutableSecondary);
  if (!created.ok) {
    return created;
  }

  const rund::AccelCheck begun = BeginCommand(adapter.device, resources.command,
                                              CommandKind::ImmutableSecondary);
  if (!begun.ok) {
    DestroyVulkanKernelCommand(adapter, resources);
    return begun;
  }

  const rund::AccelCheck encoded =
      EncodeVulkanKernelSteps(adapter, resources, resources.command.buffer);
  if (!encoded.ok) {
    DestroyVulkanKernelCommand(adapter, resources);
    return encoded;
  }
  const rund::AccelCheck ended = EndCommand(resources.command);
  if (!ended.ok) {
    DestroyVulkanKernelCommand(adapter, resources);
    return ended;
  }
  return rund::AccelCheck{true, "ok"};
}

rund::AccelCheck ExecuteVulkanKernel(VulkanAdapter &adapter,
                                     const VulkanKernelResources &resources) {
  if (adapter.command_buffer == VK_NULL_HANDLE ||
      resources.command.buffer == VK_NULL_HANDLE) {
    return rund::AccelCheck{false, "accel_vulkan_command_unavailable"};
  }
  vkCmdExecuteCommands(adapter.command_buffer, 1u, &resources.command.buffer);
  return rund::AccelCheck{true, "ok"};
}

#endif

} // namespace rund::node::accel::detail

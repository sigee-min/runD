#include "internal.hpp"

#include "../generated_indirect/internal.hpp"
#include "../generated_indirect/map.hpp"

#include "../../../../map/api.hpp"
#include "../../../../map/local.hpp"
#include "../../../local.hpp"
#include "../../prepare/record.hpp"

#include <array>

namespace rund::node::accel::detail::vulkan_sliding_detail {

#if defined(RUND_NODE_HAVE_VULKAN_SDK)
namespace {

void encode_gate_barriers(const VkCommandBuffer command,
                          const VulkanResidencySelection &selection) noexcept {
  const VulkanResidencySlidingGate &gate = selection.sliding;
  std::array<VkBufferMemoryBarrier, 4u> acquire{};
  const auto fill = [](VkBufferMemoryBarrier &barrier,
                       const VulkanBuffer &buffer, const VkAccessFlags source,
                       const VkAccessFlags target) noexcept {
    barrier.sType = VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER;
    barrier.srcAccessMask = source;
    barrier.dstAccessMask = target;
    barrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    barrier.buffer = buffer.buffer;
    barrier.offset = buffer.offset;
    barrier.size = buffer.bytes;
  };
  fill(acquire[0u], gate.descriptor,
       VK_ACCESS_HOST_WRITE_BIT | VK_ACCESS_SHADER_WRITE_BIT,
       VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT);
  fill(acquire[1u], gate.original_arguments, VK_ACCESS_HOST_WRITE_BIT,
       VK_ACCESS_SHADER_READ_BIT);
  fill(acquire[2u], gate.argument_owners, VK_ACCESS_HOST_WRITE_BIT,
       VK_ACCESS_SHADER_READ_BIT);
  fill(acquire[3u], selection.arguments,
       VK_ACCESS_HOST_WRITE_BIT | VK_ACCESS_INDIRECT_COMMAND_READ_BIT |
           VK_ACCESS_SHADER_WRITE_BIT,
       VK_ACCESS_SHADER_WRITE_BIT);
  vkCmdPipelineBarrier(command,
                       VK_PIPELINE_STAGE_HOST_BIT |
                           VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT |
                           VK_PIPELINE_STAGE_DRAW_INDIRECT_BIT,
                       VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0u, 0u, nullptr,
                       4u, acquire.data(), 0u, nullptr);
}

} // namespace

bool record_gate_command(VulkanPipeline &pipeline,
                         VulkanResidencySelection &selection) noexcept {
  VulkanResidencySlidingGate &gate = selection.sliding;
  rund::AccelCheck recorded = CreateCommand(
      pipeline.adapter->device, pipeline.adapter->compute_queue_family,
      gate.command, CommandKind::ReusablePrimary);
  if (recorded.ok) {
    recorded = BeginCommand(pipeline.adapter->device, gate.command,
                            CommandKind::ReusablePrimary);
  }
  if (!recorded.ok) {
    return false;
  }
  encode_gate_barriers(gate.command.buffer, selection);
  VkBufferMemoryBarrier control_acquire{};
  control_acquire.sType = VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER;
  control_acquire.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
  control_acquire.dstAccessMask =
      VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT;
  control_acquire.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
  control_acquire.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
  control_acquire.buffer = pipeline.control.summary.buffer;
  control_acquire.offset = pipeline.control.summary.offset;
  control_acquire.size = pipeline.control.summary.bytes;
  vkCmdPipelineBarrier(gate.command.buffer,
                       VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                       VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0u, 0u, nullptr,
                       1u, &control_acquire, 0u, nullptr);
  vkCmdBindPipeline(gate.command.buffer, VK_PIPELINE_BIND_POINT_COMPUTE,
                    gate.pipeline->pipeline);
  vkCmdBindDescriptorSets(gate.command.buffer, VK_PIPELINE_BIND_POINT_COMPUTE,
                          gate.pipeline->pipeline_layout, 0u, 1u,
                          &gate.descriptor_set, 0u, nullptr);
  vkCmdDispatch(gate.command.buffer, 1u, 1u, 1u);

  std::array<VkBufferMemoryBarrier, 3u> release{};
  const auto fill = [](VkBufferMemoryBarrier &barrier,
                       const VulkanBuffer &buffer,
                       const VkAccessFlags target) noexcept {
    barrier.sType = VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER;
    barrier.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
    barrier.dstAccessMask = target;
    barrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    barrier.buffer = buffer.buffer;
    barrier.offset = buffer.offset;
    barrier.size = buffer.bytes;
  };
  fill(release[0u], selection.arguments,
       VK_ACCESS_INDIRECT_COMMAND_READ_BIT | VK_ACCESS_SHADER_READ_BIT);
  fill(release[1u], gate.descriptor, VK_ACCESS_HOST_READ_BIT);
  fill(release[2u], pipeline.control.summary,
       VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT);
  vkCmdPipelineBarrier(
      gate.command.buffer, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
      VK_PIPELINE_STAGE_DRAW_INDIRECT_BIT |
          VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT | VK_PIPELINE_STAGE_HOST_BIT,
      0u, 0u, nullptr, static_cast<std::uint32_t>(release.size()),
      release.data(), 0u, nullptr);
  return EndCommand(gate.command).ok;
}

bool create_ready_semaphore(VulkanPipeline &pipeline,
                            VulkanResidencySlidingGate &gate) noexcept {
  VkSemaphoreTypeCreateInfoKHR type{};
  type.sType = VK_STRUCTURE_TYPE_SEMAPHORE_TYPE_CREATE_INFO_KHR;
  type.semaphoreType = VK_SEMAPHORE_TYPE_TIMELINE_KHR;
  type.initialValue = 0u;
  VkSemaphoreCreateInfo create{};
  create.sType = VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO;
  create.pNext = &type;
  return vkCreateSemaphore(pipeline.adapter->device, &create, nullptr,
                           &gate.ready) == VK_SUCCESS &&
         gate.ready != VK_NULL_HANDLE;
}

bool record_generated_map_role(
    VulkanPipeline &pipeline, VulkanResidencySelection &selection,
    const PersistentResidencySlidingRole &source, const std::uint32_t stride,
    const std::uint64_t plan_identity, const std::uint64_t token,
    const std::uint64_t run_generation,
    const std::span<const std::uint32_t> locals,
    const std::span<VkCommandBuffer> commands, std::size_t &command_count,
    std::uint64_t &dispatch_count, std::uint64_t &control_count,
    std::uint64_t &reset_count, std::uint64_t &reset_bytes,
    VulkanResidencyPersistentRun *const expected_run,
    const bool tail) noexcept {
  return vulkan_generated_indirect_detail::record_role(
      pipeline, selection, source, stride, plan_identity, token, run_generation,
      locals, commands, command_count, dispatch_count, control_count,
      reset_count, reset_bytes, expected_run, tail);
}

#endif

} // namespace rund::node::accel::detail::vulkan_sliding_detail

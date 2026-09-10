#include "control.hpp"

#include "../../command/dispatch.hpp"
#include "../control.hpp"
#include "../../../kernel/backend/run.hpp"

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>

namespace rund::node::accel::detail {

#if defined(RUND_NODE_HAVE_VULKAN_SDK)

VulkanMapControlPush VulkanMapControlParameters(
    const VulkanMapEncodeResources &map,
    const rund::kernel::ComputeDispatchWindow &window,
    const std::uint32_t index) noexcept {
  const auto split = [](const std::uint64_t value) {
    return std::array<std::uint32_t, 2u>{
        static_cast<std::uint32_t>(value),
        static_cast<std::uint32_t>(value >> 32u)};
  };
  const std::uint64_t capacity =
      map.control.capacity == 0u
          ? map.windows.back().begin_sequence + map.windows.back().tile_count
          : map.control.capacity;
  const auto cap = split(capacity);
  const auto expected = split(map.control.predicate_expected);
  const auto begin = split(window.begin_sequence);
  const auto count = split(window.tile_count);
  const std::uint64_t count_offset =
      map.control_count.ref.offset_bytes + map.control.count_byte_offset;
  const std::uint64_t predicate_offset =
      map.control_predicate.ref.offset_bytes +
      map.control.predicate_byte_offset;
  VulkanMapControlPush params{};
  params.words = {
      map.control.has_count() ? 1u : 0u,
      map.control.count_source == rund::kernel::GraphControlSource::U64 ? 1u
                                                                        : 0u,
      map.control.has_predicate() ? 1u : 0u,
      map.control.predicate_source == rund::kernel::GraphControlSource::U64
          ? 1u
          : 0u,
      cap[0],
      cap[1],
      expected[0],
      expected[1],
      begin[0],
      begin[1],
      count[0],
      count[1],
      index,
      static_cast<std::uint32_t>((count_offset - map.count_base) /
                                 sizeof(std::uint32_t)),
      static_cast<std::uint32_t>((predicate_offset - map.predicate_base) /
                                 sizeof(std::uint32_t)),
      map.prepared->checks.empty() ? 0u : 1u};
  if (map.mode == VulkanMapMode::Generated) {
    params.words[15] = 0x80000000u;
  }
  return params;
}

void EncodeVulkanMapCheck(VkCommandBuffer command,
                          const VulkanMapEncodeResources &map) {
  if (map.prepared == nullptr || map.prepared->checks.empty()) {
    return;
  }
  const VulkanMapControlPush params =
      VulkanMapControlParameters(map, map.windows.front(), 0u);
  BindVulkanPipeline(command, VK_PIPELINE_BIND_POINT_COMPUTE,
                     map.check_pipeline->pipeline);
  BindVulkanDescriptors(command, VK_PIPELINE_BIND_POINT_COMPUTE,
                        map.check_pipeline->pipeline_layout, 0u, 1u,
                        &map.check_descriptor, 0u, nullptr);
  PushVulkanConstants(command, map.check_pipeline->pipeline_layout,
                      VK_SHADER_STAGE_COMPUTE_BIT, 0u, sizeof(params), &params);
  DispatchVulkan(command, 1u, 1u, 1u);
  const VkBufferMemoryBarrier barrier{
      .sType = VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER,
      .pNext = nullptr,
      .srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT,
      .dstAccessMask = VK_ACCESS_SHADER_READ_BIT,
      .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
      .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
      .buffer = map.control_status.device.buffer,
      .offset = 0u,
      .size = 2u * sizeof(std::uint32_t),
  };
  vkCmdPipelineBarrier(command, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                       VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0u, 0u, nullptr,
                       1u, &barrier, 0u, nullptr);
}

void EncodeVulkanMapControl(
    VkCommandBuffer command, const VulkanMapEncodeResources &map,
    const rund::kernel::ComputeDispatchWindow &window,
    const std::uint32_t index) {
  const VulkanMapControlPush base =
      VulkanMapControlParameters(map, window, index);
  if (map.mode == VulkanMapMode::Generated) {
    VulkanGeneratedMapControlPush params{};
    std::copy(base.words.begin(), base.words.end(), params.words.begin());
    const auto split = [](const std::uint64_t value) {
      return std::array<std::uint32_t, 2u>{
          static_cast<std::uint32_t>(value),
          static_cast<std::uint32_t>(value >> 32u)};
    };
    const auto owner = split(map.generated_owner);
    const auto plan = split(map.generated_plan);
    const auto token = split(map.generated_token);
    const auto run = split(map.generated_run);
    const auto descriptor = split(map.generated_first_descriptor_generation);
    const auto descriptor_stride =
        split(map.generated_descriptor_generation_stride);
    params.words[16] = owner[0];
    params.words[17] = owner[1];
    params.words[18] = plan[0];
    params.words[19] = plan[1];
    params.words[20] = token[0];
    params.words[21] = token[1];
    params.words[22] = run[0];
    params.words[23] = run[1];
    params.words[24] = map.generated_slot;
    params.words[25] = map.generated_stride;
    params.words[26] = map.generated_first_control_generation;
    params.words[27] = map.generated_control_generation_stride;
    params.words[28] = descriptor[0];
    params.words[29] = descriptor[1];
    params.words[30] = descriptor_stride[0];
    params.words[31] = descriptor_stride[1];
    std::array<VkBufferMemoryBarrier, 2u> acquire{};
    const auto fill = [](VkBufferMemoryBarrier &barrier,
                         const VulkanBuffer &buffer) noexcept {
      barrier.sType = VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER;
      barrier.srcAccessMask =
          VK_ACCESS_HOST_WRITE_BIT | VK_ACCESS_SHADER_WRITE_BIT;
      barrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
      barrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
      barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
      barrier.buffer = buffer.buffer;
      barrier.offset = buffer.offset;
      barrier.size = buffer.bytes;
    };
    fill(acquire[0u], map.generated_gate_binding);
    fill(acquire[1u], map.generated_summary_binding);
    vkCmdPipelineBarrier(command,
                         VK_PIPELINE_STAGE_HOST_BIT |
                             VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                         VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0u, 0u, nullptr,
                         2u, acquire.data(), 0u, nullptr);
    BindVulkanPipeline(command, VK_PIPELINE_BIND_POINT_COMPUTE,
                       map.generated_control_pipeline->pipeline);
    BindVulkanDescriptors(command, VK_PIPELINE_BIND_POINT_COMPUTE,
                          map.generated_control_pipeline->pipeline_layout, 0u,
                          1u, &map.generated_control_descriptor, 0u, nullptr);
    PushVulkanConstants(
        command, map.generated_control_pipeline->pipeline_layout,
        VK_SHADER_STAGE_COMPUTE_BIT, 0u, sizeof(params), &params);
    DispatchVulkan(command, 1u, 1u, 1u);
    const VkBufferMemoryBarrier release{
        .sType = VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER,
        .pNext = nullptr,
        .srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT,
        .dstAccessMask =
            VK_ACCESS_INDIRECT_COMMAND_READ_BIT | VK_ACCESS_HOST_READ_BIT,
        .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
        .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
        .buffer = map.generated_gate_binding.buffer,
        .offset = map.generated_gate_binding.offset,
        .size = map.generated_gate_binding.bytes,
    };
    const VkBufferMemoryBarrier args_release{
        .sType = VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER,
        .pNext = nullptr,
        .srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT,
        .dstAccessMask = VK_ACCESS_INDIRECT_COMMAND_READ_BIT,
        .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
        .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
        .buffer = map.control_args.buffer.buffer,
        .offset = index * 4u * sizeof(std::uint32_t),
        .size = 4u * sizeof(std::uint32_t),
    };
    const std::array<VkBufferMemoryBarrier, 2u> releases{release, args_release};
    vkCmdPipelineBarrier(
        command, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
        VK_PIPELINE_STAGE_DRAW_INDIRECT_BIT | VK_PIPELINE_STAGE_HOST_BIT, 0u,
        0u, nullptr, static_cast<std::uint32_t>(releases.size()),
        releases.data(), 0u, nullptr);
    return;
  }
  const VulkanMapControlPush &params = base;
  BindVulkanPipeline(command, VK_PIPELINE_BIND_POINT_COMPUTE,
                     map.control_pipeline->pipeline);
  BindVulkanDescriptors(command, VK_PIPELINE_BIND_POINT_COMPUTE,
                        map.control_pipeline->pipeline_layout, 0u, 1u,
                        &map.control_descriptor, 0u, nullptr);
  PushVulkanConstants(command, map.control_pipeline->pipeline_layout,
                      VK_SHADER_STAGE_COMPUTE_BIT, 0u, sizeof(params), &params);
  DispatchVulkan(command, 1u, 1u, 1u);
  const VkBufferMemoryBarrier barrier{
      .sType = VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER,
      .pNext = nullptr,
      .srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT,
      .dstAccessMask = VK_ACCESS_INDIRECT_COMMAND_READ_BIT,
      .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
      .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
      .buffer = map.control_args.buffer.buffer,
      .offset = index * 4u * sizeof(std::uint32_t),
      .size = 4u * sizeof(std::uint32_t),
  };
  vkCmdPipelineBarrier(command, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                       VK_PIPELINE_STAGE_DRAW_INDIRECT_BIT, 0u, 0u, nullptr, 1u,
                       &barrier, 0u, nullptr);
}

#endif

} // namespace rund::node::accel::detail

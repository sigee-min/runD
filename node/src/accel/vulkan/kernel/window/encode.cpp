#include "internal.hpp"

#include "../../command/capture.hpp"
#include "../../command/dispatch.hpp"
#include "../../command.hpp"
#include "../../command/timestamp.hpp"
#include "../../descriptor.hpp"

#include "../../../kernel/backend/exception.hpp"

#include <algorithm>
#include <array>
#include <limits>

namespace rund::node::accel::detail {

#if defined(RUND_NODE_HAVE_VULKAN_SDK)
namespace {

struct GateParams final {
  std::uint32_t state{};
  std::uint32_t source_word{};
  std::uint32_t target_word{};
};

static_assert(sizeof(GateParams) == VulkanGateParameterBytes);

void EncodeDispatchBarrier(const VkCommandBuffer command) noexcept {
  VkMemoryBarrier barrier{};
  barrier.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER;
  barrier.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
  barrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT |
                          VK_ACCESS_SHADER_WRITE_BIT |
                          VK_ACCESS_INDIRECT_COMMAND_READ_BIT;
  vkCmdPipelineBarrier(command, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                       VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT |
                           VK_PIPELINE_STAGE_DRAW_INDIRECT_BIT,
                       0u, 1u, &barrier, 0u, nullptr, 0u, nullptr);
}

} // namespace

bool EncodeVulkanWindowIndirect(void *const context,
                                VulkanDispatchCapture &capture,
                                const VkCommandBuffer command,
                                const VkBuffer source,
                                const VkDeviceSize offset) noexcept {
  auto *const resources = static_cast<VulkanWindowResources *>(context);
  if (resources == nullptr || resources->adapter == nullptr ||
      resources->gate_pipeline == nullptr || command == VK_NULL_HANDLE ||
      source == VK_NULL_HANDLE || (offset & 3u) != 0u ||
      capture.mapped == nullptr || capture.original == nullptr ||
      capture.owners == nullptr || capture.arguments == VK_NULL_HANDLE ||
      capture.cursor >= capture.capacity ||
      capture.pipeline == VK_NULL_HANDLE || capture.layout == VK_NULL_HANDLE ||
      capture.descriptor == VK_NULL_HANDLE ||
      resources->adapter->storage_align == 0u ||
      (!capture.replay &&
       resources->gates.size() >= resources->gate_capacity) ||
      (capture.replay && capture.indirect_count >= resources->gates.size())) {
    return false;
  }
  const VkDeviceSize base =
      offset - (offset % resources->adapter->storage_align);
  if (offset > std::numeric_limits<VkDeviceSize>::max() -
                   sizeof(VkDispatchIndirectCommand)) {
    return false;
  }
  const VkDeviceSize end = offset + sizeof(VkDispatchIndirectCommand);
  const VkDeviceSize range = end - base;
  if (range == 0u || range > resources->adapter->storage_limit) {
    return false;
  }
  VulkanGateRoute *route = nullptr;
  if (capture.replay) {
    route = &resources->gates[capture.indirect_count];
    if (route->source.buffer != source || route->base != base ||
        route->offset != offset) {
      return false;
    }
  } else {
    try {
      resources->gates.push_back(VulkanGateRoute{
          .source =
              VulkanBuffer{
                  .buffer = source,
                  .bytes = end,
                  .usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT |
                           VK_BUFFER_USAGE_INDIRECT_BUFFER_BIT,
              },
          .base = base,
          .offset = offset,
      });
    } catch (...) {
      backend_exception::RethrowUnlessCapacityException();
      return false;
    }
    route = &resources->gates.back();
    if (!AcquireVulkanCollectiveDescriptorSet(*resources->adapter,
                                              *resources->gate_pipeline, 3u,
                                              route->descriptor)) {
      resources->gates.pop_back();
      return false;
    }
    const std::array<VulkanStorageBinding, 3u> bindings{
        VulkanStorageBinding{&route->source, base, range},
        VulkanStorageBindingFor(resources->arguments),
        VulkanStorageBindingFor(resources->states),
    };
    if (!WriteVulkanStorageDescriptorSet(*resources->adapter, route->descriptor,
                                         bindings)) {
      resources->gates.pop_back();
      return false;
    }
  }
  const std::size_t slot = capture.cursor++;
  if (capture.replay) {
    if (capture.original[slot].x != 0u || capture.original[slot].y != 0u ||
        capture.original[slot].z != 0u ||
        capture.owners[slot] != std::numeric_limits<std::uint32_t>::max()) {
      return false;
    }
  } else {
    capture.mapped[slot] = {};
    capture.original[slot] = {};
    capture.owners[slot] = std::numeric_limits<std::uint32_t>::max();
  }
  const GateParams params{
      .state = capture.owner,
      .source_word = static_cast<std::uint32_t>((offset - base) / 4u),
      .target_word = static_cast<std::uint32_t>(slot * 3u),
  };
  EncodeVulkanComputeToComputeBarrier(command);
  ::vkCmdBindPipeline(command, VK_PIPELINE_BIND_POINT_COMPUTE,
                      resources->gate_pipeline->pipeline);
  ::vkCmdBindDescriptorSets(command, VK_PIPELINE_BIND_POINT_COMPUTE,
                            resources->gate_pipeline->pipeline_layout, 0u, 1u,
                            &route->descriptor, 0u, nullptr);
  ::vkCmdPushConstants(command, resources->gate_pipeline->pipeline_layout,
                       VK_SHADER_STAGE_COMPUTE_BIT, 0u, sizeof(params),
                       &params);
  ::vkCmdDispatch(command, 1u, 1u, 1u);
  EncodeDispatchBarrier(command);
  ::vkCmdBindPipeline(command, VK_PIPELINE_BIND_POINT_COMPUTE,
                      capture.pipeline);
  ::vkCmdBindDescriptorSets(command, VK_PIPELINE_BIND_POINT_COMPUTE,
                            capture.layout, 0u, 1u, &capture.descriptor, 0u,
                            nullptr);
  if (capture.has_push) {
    ::vkCmdPushConstants(command, capture.layout, capture.push_stages,
                         capture.push_offset, capture.push_size,
                         capture.push.data() + capture.push_offset);
  }
  WriteVulkanDispatchTimestamp(command);
  ::vkCmdDispatchIndirect(
      command, capture.arguments,
      static_cast<VkDeviceSize>(slot * sizeof(VkDispatchIndirectCommand)));
  WriteVulkanDispatchTimestamp(command);
  ++capture.indirect_count;
  return true;
}

bool EncodeVulkanWindow(const VkCommandBuffer command,
                        const VulkanWindowResources &resources,
                        const std::uint32_t entry,
                        const bool preflight) noexcept {
  const auto begin = std::lower_bound(
      resources.routes.begin(), resources.routes.end(), entry,
      [](const VulkanWindowRoute &route, const std::uint32_t value) {
        return route.entry < value;
      });
  const auto end = std::upper_bound(
      begin, resources.routes.end(), entry,
      [](const std::uint32_t value, const VulkanWindowRoute &route) {
        return value < route.entry;
      });
  if (begin == end) {
    return true;
  }
  if (command == VK_NULL_HANDLE || resources.pipeline == nullptr) {
    return false;
  }
  EncodeVulkanComputeToComputeBarrier(command);
  BindVulkanPipeline(command, VK_PIPELINE_BIND_POINT_COMPUTE,
                     resources.pipeline->pipeline);
  for (auto route = begin; route != end; ++route) {
    BackendWindowPhase phase{};
    bool stored_preflight = false;
    if (route->descriptor == VK_NULL_HANDLE ||
        !DecodeBackendWindowParameter(route->params.phase, phase,
                                      stored_preflight) ||
        stored_preflight ||
        (preflight && phase != BackendWindowPhase::NestedSeed)) {
      return false;
    }
    if (!preflight && phase == BackendWindowPhase::NestedAction &&
        route->params.inner_advance == 0u) {
      continue;
    }
    VulkanWindowParams params = route->params;
    if (!EncodeBackendWindowParameter(phase, preflight, params.phase)) {
      return false;
    }
    BindVulkanDescriptors(command, VK_PIPELINE_BIND_POINT_COMPUTE,
                          resources.pipeline->pipeline_layout, 0u, 1u,
                          &route->descriptor, 0u, nullptr);
    PushVulkanConstants(command, resources.pipeline->pipeline_layout,
                        VK_SHADER_STAGE_COMPUTE_BIT, 0u, sizeof(params),
                        &params);
    ::vkCmdDispatch(command, 1u, 1u, 1u);
  }
  EncodeDispatchBarrier(command);
  return true;
}

bool EncodeVulkanWindowStart(const VkCommandBuffer command,
                             const VulkanWindowResources &resources) noexcept {
  if (command == VK_NULL_HANDLE || resources.states.buffer == VK_NULL_HANDLE ||
      resources.arguments.buffer == VK_NULL_HANDLE ||
      resources.original_arguments.buffer == VK_NULL_HANDLE ||
      resources.states.bytes == 0u || resources.arguments.bytes == 0u ||
      resources.arguments.bytes != resources.original_arguments.bytes) {
    return false;
  }
  vkCmdFillBuffer(command, resources.states.buffer, 0u, resources.states.bytes,
                  0u);
  VkBufferCopy copy{};
  copy.size = resources.arguments.bytes;
  vkCmdCopyBuffer(command, resources.original_arguments.buffer,
                  resources.arguments.buffer, 1u, &copy);
  VkMemoryBarrier barrier{};
  barrier.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER;
  barrier.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
  barrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT |
                          VK_ACCESS_SHADER_WRITE_BIT |
                          VK_ACCESS_INDIRECT_COMMAND_READ_BIT;
  vkCmdPipelineBarrier(command, VK_PIPELINE_STAGE_TRANSFER_BIT,
                       VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT |
                           VK_PIPELINE_STAGE_DRAW_INDIRECT_BIT,
                       0u, 1u, &barrier, 0u, nullptr, 0u, nullptr);
  return true;
}

#endif

} // namespace rund::node::accel::detail

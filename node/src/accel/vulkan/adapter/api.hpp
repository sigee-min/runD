#pragma once

#include <accel/check.hpp>
#include <accel/device.hpp>
#include <accel/runtime.hpp>

#include "../../backend/result.hpp"
#include "shader.hpp"
#include "state.hpp"

#include <array>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <span>

namespace rund::node::accel::detail {

#if defined(RUND_NODE_HAVE_VULKAN_SDK)

struct ResidentDesc;
struct VulkanResidentBufferResult;

struct VulkanDispatchCapture final {
  using Indirect = bool (*)(void *, VulkanDispatchCapture &, VkCommandBuffer,
                            VkBuffer, VkDeviceSize) noexcept;

  VkCommandBuffer command{VK_NULL_HANDLE};
  VkBuffer arguments{VK_NULL_HANDLE};
  VkDispatchIndirectCommand *mapped{};
  VkDispatchIndirectCommand *original{};
  std::uint32_t *owners{};
  std::size_t capacity{};
  std::size_t cursor{};
  std::uint32_t owner{};
  VkPipeline pipeline{VK_NULL_HANDLE};
  VkPipelineLayout layout{VK_NULL_HANDLE};
  VkDescriptorSet descriptor{VK_NULL_HANDLE};
  VkShaderStageFlags push_stages{};
  std::uint32_t push_offset{};
  std::uint32_t push_size{};
  std::array<std::byte, 256u> push{};
  void *context{};
  Indirect indirect{};
  std::uint64_t indirect_count{};
  bool has_push{};
  bool failed{};
};

inline thread_local VulkanDispatchCapture *vulkan_dispatch_capture = nullptr;

[[nodiscard]] inline bool
CapturesVulkanDispatch(const VkCommandBuffer command) noexcept {
  return vulkan_dispatch_capture != nullptr &&
         vulkan_dispatch_capture->command == command;
}

class VulkanDispatchScope final {
public:
  VulkanDispatchScope(VulkanDispatchCapture &capture,
                      const VkCommandBuffer command,
                      const std::uint32_t owner) noexcept
      : previous_(vulkan_dispatch_capture) {
    capture.command = command;
    capture.owner = owner;
    vulkan_dispatch_capture = &capture;
  }

  ~VulkanDispatchScope() { vulkan_dispatch_capture = previous_; }

  VulkanDispatchScope(const VulkanDispatchScope &) = delete;
  VulkanDispatchScope &operator=(const VulkanDispatchScope &) = delete;

private:
  VulkanDispatchCapture *previous_{};
};

inline void BindVulkanPipeline(const VkCommandBuffer command,
                               const VkPipelineBindPoint point,
                               const VkPipeline pipeline) noexcept {
  ::vkCmdBindPipeline(command, point, pipeline);
  VulkanDispatchCapture *const capture = vulkan_dispatch_capture;
  if (capture == nullptr || capture->command != command ||
      point != VK_PIPELINE_BIND_POINT_COMPUTE) {
    return;
  }
  capture->pipeline = pipeline;
  capture->layout = VK_NULL_HANDLE;
  capture->descriptor = VK_NULL_HANDLE;
  capture->has_push = false;
}

inline void BindVulkanDescriptors(
    const VkCommandBuffer command, const VkPipelineBindPoint point,
    const VkPipelineLayout layout, const std::uint32_t first_set,
    const std::uint32_t set_count, const VkDescriptorSet *const sets,
    const std::uint32_t dynamic_count,
    const std::uint32_t *const dynamic_offsets) noexcept {
  ::vkCmdBindDescriptorSets(command, point, layout, first_set, set_count, sets,
                            dynamic_count, dynamic_offsets);
  VulkanDispatchCapture *const capture = vulkan_dispatch_capture;
  if (capture == nullptr || capture->command != command ||
      point != VK_PIPELINE_BIND_POINT_COMPUTE) {
    return;
  }
  if (first_set != 0u || set_count != 1u || sets == nullptr ||
      sets[0] == VK_NULL_HANDLE || dynamic_count != 0u ||
      dynamic_offsets != nullptr) {
    capture->failed = true;
    return;
  }
  capture->layout = layout;
  capture->descriptor = sets[0];
}

inline void PushVulkanConstants(const VkCommandBuffer command,
                                const VkPipelineLayout layout,
                                const VkShaderStageFlags stages,
                                const std::uint32_t offset,
                                const std::uint32_t size,
                                const void *const data) noexcept {
  ::vkCmdPushConstants(command, layout, stages, offset, size, data);
  VulkanDispatchCapture *const capture = vulkan_dispatch_capture;
  if (capture == nullptr || capture->command != command) {
    return;
  }
  if (data == nullptr || offset > capture->push.size() ||
      size > capture->push.size() - offset) {
    capture->failed = true;
    return;
  }
  std::memcpy(capture->push.data() + offset, data, size);
  capture->push_stages = stages;
  capture->push_offset = offset;
  capture->push_size = size;
  capture->has_push = true;
}

// Pipeline preparation redirects fixed Program dispatches through one
// immutable indirect-argument arena. Outside a VulkanDispatchScope this is
// exactly the native command.
inline void DispatchVulkan(const VkCommandBuffer command, const std::uint32_t x,
                           const std::uint32_t y,
                           const std::uint32_t z) noexcept {
  VulkanDispatchCapture *const capture = vulkan_dispatch_capture;
  if (capture == nullptr || capture->command != command) {
    ::vkCmdDispatch(command, x, y, z);
    return;
  }
  if (capture->mapped == nullptr || capture->original == nullptr ||
      capture->owners == nullptr || capture->arguments == VK_NULL_HANDLE ||
      capture->cursor >= capture->capacity) {
    capture->failed = true;
    return;
  }
  const std::size_t slot = capture->cursor++;
  const VkDispatchIndirectCommand value{
      .x = x,
      .y = y,
      .z = z,
  };
  capture->mapped[slot] = value;
  capture->original[slot] = value;
  capture->owners[slot] = capture->owner;
  ::vkCmdDispatchIndirect(
      command, capture->arguments,
      static_cast<VkDeviceSize>(slot * sizeof(VkDispatchIndirectCommand)));
}

inline void DispatchVulkanIndirect(const VkCommandBuffer command,
                                   const VkBuffer source,
                                   const VkDeviceSize offset) noexcept {
  VulkanDispatchCapture *const capture = vulkan_dispatch_capture;
  if (capture == nullptr || capture->command != command) {
    ::vkCmdDispatchIndirect(command, source, offset);
    return;
  }
  if (capture->indirect == nullptr ||
      !capture->indirect(capture->context, *capture, command, source, offset)) {
    capture->failed = true;
  }
}

void SetVulkanLastError(VulkanAdapter &adapter, const char *reason);

[[nodiscard]] constexpr const char *
VulkanFailureReason(const VkResult result,
                    const char *const fallback) noexcept {
  return result == VK_ERROR_DEVICE_LOST ? "compute_device_lost" : fallback;
}

[[nodiscard]] bool CompileVulkanShader(
    VulkanAdapter &adapter, const rund::kernel::ComputePlan &plan,
    const rund::kernel::LoweringArtifact &artifact, VulkanShader &shader);

[[nodiscard]] bool CompileVulkanSourceWithTools(
    VulkanAdapter &adapter, const rund::kernel::ComputePlan &plan,
    const rund::kernel::LoweringArtifact &artifact, VulkanShader &shader);

[[nodiscard]] bool
CreateVulkanBuffer(VulkanAdapter &adapter, VkDeviceSize bytes,
                   VkBufferUsageFlags usage, VulkanBuffer &buffer,
                   bool *reused = nullptr,
                   VulkanMemoryUse memory_use = VulkanMemoryUse::Staging);

void DestroyVulkanBuffer(VulkanAdapter &adapter, VulkanBuffer &buffer);

void ReleaseVulkanBuffer(VulkanAdapter &adapter, VulkanBuffer &buffer);

[[nodiscard]] bool UploadVulkanBuffer(VulkanBuffer &buffer, const void *data,
                                      VkDeviceSize bytes);

[[nodiscard]] bool ClearVulkanBuffer(VulkanBuffer &buffer, VkDeviceSize bytes);

[[nodiscard]] bool
VulkanPickOwnsAdapter(const rund::AccelDevice &pick) noexcept;

[[nodiscard]] VulkanAdapter *
CheckedVulkanAdapter(const rund::AccelDevice &pick) noexcept;

[[nodiscard]] VulkanResidentBufferResult
CreateVulkanResidentBuffer(const rund::AccelDevice &pick,
                           const ResidentDesc &desc,
                           bool zero_initialize = false);

[[nodiscard]] VulkanResidentBufferResult
LookupVulkanResidentBuffer(const rund::AccelDevice &pick,
                           const rund::kernel::ResidentBufferRef &ref,
                           const std::shared_ptr<void> &handle);

[[nodiscard]] rund::AccelCheck UploadVulkanResidentBuffer(
    const rund::AccelDevice &pick, const rund::kernel::ResidentBufferRef &ref,
    const std::shared_ptr<void> &handle, const void *data,
    rund::kernel::u64 bytes, rund::kernel::u64 offset);

[[nodiscard]] BackendDownload DownloadVulkanResidentBuffer(
    const rund::AccelDevice &pick, const rund::kernel::ResidentBufferRef &ref,
    const std::shared_ptr<void> &handle, void *data, rund::kernel::u64 bytes,
    rund::kernel::u64 offset, bool hash_payload);

[[nodiscard]] BackendUpload
UploadVulkanResidentBuffers(const rund::AccelDevice &pick,
                            std::span<const UploadRoute> requests,
                            TransferCompletion completion);

[[nodiscard]] BackendDownload
DownloadVulkanResidentBuffers(const rund::AccelDevice &pick,
                              std::span<const DownloadRoute> requests);

[[nodiscard]] BackendCopy
CopyVulkanResidentBuffers(const rund::AccelDevice &pick,
                          std::span<const CopyRoute> requests);

[[nodiscard]] rund::RuntimeStats
ReadVulkanRuntimeStats(const rund::AccelDevice &pick);
void ResetVulkanRuntimeStats(const rund::AccelDevice &pick);

[[nodiscard]] bool
ExecuteVulkan(void *context, const rund::kernel::ComputePlan &plan,
              const rund::kernel::LoweringArtifact &artifact,
              const rund::kernel::ComputeDispatchWindow *windows,
              rund::kernel::u64 window_count,
              const rund::kernel::BindingSet &bindings);

[[nodiscard]] const char *VulkanLastError(void *context);

#endif // defined(RUND_NODE_HAVE_VULKAN_SDK)

} // namespace rund::node::accel::detail

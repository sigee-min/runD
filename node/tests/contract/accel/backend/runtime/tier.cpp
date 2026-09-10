#include "local.hpp"

#include "src/accel/backend/token.hpp"
#include "src/accel/kernel/preparation.hpp"
#include "src/accel/vulkan/adapter/access.hpp"
#include "src/accel/vulkan/buffer/access.hpp"
#include "src/accel/vulkan/buffer/create.hpp"
#include "src/accel/vulkan/buffer/resident/lookup.hpp"
#include "src/accel/vulkan/command/resources.hpp"
#include "src/accel/vulkan/resident/access.hpp"
#include "src/accel/vulkan/status.hpp"
#include <node/accel/buffer.hpp>
#include <node/accel/pick.hpp>

#include <accel/runtime.hpp>

#include <array>
#include <memory>
#include <mutex>

namespace node_accel_contract::backend_runtime {
namespace {

namespace detail = rund::node::accel::detail;

} // namespace

bool CheckVulkanMemoryTier(const rund::AccelDevice &pick) {
  if (!pick.check.ok) {
    return node_accel_contract::vulkan::FailureReasonIsPrecise(pick);
  }
#if defined(RUND_NODE_HAVE_VULKAN_SDK)
  const std::shared_ptr<detail::PickToken> token = detail::AdmitPick(pick);
  const rund::AccelDevice *const raw = token == nullptr ? nullptr : &token->raw;
  detail::VulkanAdapter *const adapter =
      raw == nullptr ? nullptr : detail::CheckedVulkanAdapter(*raw);
  if (adapter == nullptr) {
    return false;
  }

  {
    const rund::Buffer buffer = rund::node::accel::CreateBuffer(
        pick, rund::BufferDesc{.bytes = 7u,
                               .usage = rund::BufferUsage::ReadWrite,
                               .alignment = 16u});
    if (!buffer.check.ok || buffer.storage_bytes != 8u) {
      return false;
    }
    const rund::kernel::ResidentBufferRef ref =
        rund::node::accel::ResidentRef(buffer);
    const detail::VulkanResidentBufferResult resident =
        detail::LookupVulkanResidentBuffer(*raw, ref, buffer.handle);
    if (!resident.check.ok || resident.device_buffer == nullptr ||
        resident.device_buffer->mapped != nullptr ||
        resident.device_buffer->memory_use !=
            detail::VulkanMemoryUse::Resident ||
        (resident.device_buffer->memory_flags &
         VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT) == 0u ||
        (resident.device_buffer->usage & VK_BUFFER_USAGE_TRANSFER_SRC_BIT) ==
            0u ||
        (resident.device_buffer->usage & VK_BUFFER_USAGE_TRANSFER_DST_BIT) ==
            0u) {
      return false;
    }

    std::array<std::uint8_t, 7u> expected{1u, 2u, 3u, 4u, 5u, 6u, 7u};
    const std::array<std::uint8_t, 3u> patch{11u, 12u, 13u};
    expected[2] = patch[0];
    expected[3] = patch[1];
    expected[4] = patch[2];
    const std::array<std::uint8_t, 7u> initial{1u, 2u, 3u, 4u, 5u, 6u, 7u};
    std::array<std::uint8_t, 7u> output{};
    rund::node::accel::ResetRuntimeStats(pick);
    if (!rund::node::accel::UploadBuffer(pick, buffer, initial.data(),
                                         initial.size())
             .ok ||
        !rund::node::accel::UploadBuffer(pick, buffer, patch.data(),
                                         patch.size(), 2u)
             .ok ||
        !rund::node::accel::DownloadBuffer(pick, buffer, output.data(),
                                           output.size())
             .ok ||
        output != expected) {
      return false;
    }
    const rund::RuntimeStats stats = rund::node::accel::ReadRuntimeStats(pick);
    if (!stats.outcome.ok || stats.run.transfer.host_to_device_bytes != 10u ||
        stats.run.transfer.device_to_host_bytes != 7u ||
        stats.run.work.command_submit_count < 4u) {
      return false;
    }
  }
  {
    auto &resident = detail::VulkanResidents(*adapter);
    std::lock_guard lock{resident.mutex};
    if (!resident.buffers.empty()) {
      return false;
    }
  }
  {
    const rund::Buffer reused = rund::node::accel::CreateBuffer(
        pick, rund::BufferDesc{.bytes = 7u,
                               .usage = rund::BufferUsage::ReadWrite,
                               .alignment = 16u});
    if (!reused.check.ok || !reused.storage_reused) {
      return false;
    }
  }

  {
    auto &resident = detail::VulkanResidents(*adapter);
    std::lock_guard lock{resident.mutex};
    if (!resident.buffers.empty()) {
      return false;
    }
  }
  std::lock_guard lock{adapter->mutex};
  detail::VulkanBuffer staging{};
  detail::VulkanBuffer device{};
  detail::VulkanStatus status{};
  detail::VulkanStatus pipeline_status{};
  const bool created =
      detail::CreateVulkanBuffer(*adapter, 16u,
                                 VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, staging,
                                 nullptr, detail::VulkanMemoryUse::Staging) &&
      detail::CreateVulkanBuffer(*adapter, 16u,
                                 VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, device,
                                 nullptr, detail::VulkanMemoryUse::Device) &&
      detail::CreateVulkanStatus(*adapter, 16u, status) &&
      detail::CreateVulkanStatus(
          *adapter, 16u, pipeline_status,
          detail::KernelPreparationMode::PipelinePrivate);
  const bool tiered =
      created && staging.mapped != nullptr && device.mapped == nullptr &&
      (staging.memory_flags & (VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT |
                               VK_MEMORY_PROPERTY_HOST_COHERENT_BIT)) ==
          (VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT |
           VK_MEMORY_PROPERTY_HOST_COHERENT_BIT) &&
      (device.memory_flags & VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT) != 0u &&
      status.device.mapped == nullptr &&
      status.device.memory_use == detail::VulkanMemoryUse::Device &&
      (status.device.memory_flags & VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT) !=
          0u &&
      (status.device.usage & VK_BUFFER_USAGE_TRANSFER_SRC_BIT) != 0u &&
      (status.device.usage & VK_BUFFER_USAGE_TRANSFER_DST_BIT) != 0u &&
      status.readback.bytes == 4u * sizeof(rund::kernel::u32) &&
      status.readback.mapped != nullptr &&
      status.readback.memory_use == detail::VulkanMemoryUse::Staging &&
      (status.readback.memory_flags & (VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT |
                                       VK_MEMORY_PROPERTY_HOST_COHERENT_BIT)) ==
          (VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT |
           VK_MEMORY_PROPERTY_HOST_COHERENT_BIT) &&
      (status.readback.usage & VK_BUFFER_USAGE_TRANSFER_DST_BIT) != 0u &&
      pipeline_status.pipeline &&
      pipeline_status.device.memory_use == detail::VulkanMemoryUse::Device &&
      pipeline_status.device.mapped == nullptr &&
      pipeline_status.readback.buffer == VK_NULL_HANDLE &&
      pipeline_status.readback.memory == VK_NULL_HANDLE &&
      pipeline_status.readback.bytes == 0u &&
      pipeline_status.readback.mapped == nullptr;
  detail::ReleaseVulkanBuffer(*adapter, staging);
  detail::ReleaseVulkanBuffer(*adapter, device);
  detail::ReleaseVulkanStatus(*adapter, status);
  detail::ReleaseVulkanStatus(*adapter, pipeline_status);
  return tiered;
#else
  return false;
#endif
}

} // namespace node_accel_contract::backend_runtime

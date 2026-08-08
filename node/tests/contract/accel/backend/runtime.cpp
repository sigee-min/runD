#include <accel/api.hpp>
#include <accel/device.hpp>
#include <accel/runtime.hpp>

#include "../metal/stats/run.hpp"
#include "../vulkan/stats/run.hpp"
#include "run/local.hpp"
#include "src/accel/backend/resource.hpp"
#include "src/accel/backend/token.hpp"
#include "src/accel/cpu/buffer.hpp"
#include "src/accel/metal/state.hpp"
#include "src/accel/vulkan/adapter/api.hpp"
#include "src/accel/vulkan/command/resources.hpp"
#include "src/accel/vulkan/resident/access.hpp"
#include "src/accel/vulkan/status.hpp"
#include <node/accel/buffer.hpp>
#include <node/accel/pick.hpp>
#include <rund/counter.hpp>

#include <array>
#include <chrono>
#include <cstdint>
#include <limits>
#include <memory>
#include <mutex>
#include <string_view>
#include <thread>
#include <utility>
#include <vector>

namespace node_accel_contract {
[[nodiscard]] bool ResidentValidationContract();
}

namespace {

using rund::AccelDevice;
namespace detail = rund::node::accel::detail;

constexpr std::uint64_t kCounterMaximum =
    std::numeric_limits<std::uint64_t>::max();

consteval bool VulkanCommandRingContract() {
  detail::VulkanCommandRing ring{};
  std::array<detail::VulkanCommandLease, detail::kVulkanCommandCapacity>
      commands{};
  for (auto &command : commands) {
    command = ring.claim();
    if (!command || !ring.publish(command)) {
      return false;
    }
  }
  if (ring.claim() || ring.retire(commands[1])) {
    return false;
  }
  for (const auto command : commands) {
    if (!ring.retire(command)) {
      return false;
    }
  }
  detail::VulkanCommandLease terminal = ring.claim();
  ring.next_sequence = kCounterMaximum;
  return ring.empty() == false && !ring.publish(terminal) &&
         ring.cancel(terminal) && ring.empty();
}

static_assert(VulkanCommandRingContract());

#if defined(RUND_NODE_HAVE_VULKAN_SDK)
consteval bool VulkanCommandResourceContract() {
  constexpr detail::CommandPlan primary =
      detail::PlanCommand(detail::CommandKind::OneShotPrimary);
  constexpr detail::CommandPlan secondary =
      detail::PlanCommand(detail::CommandKind::ImmutableSecondary);
  constexpr detail::CommandPlan reusable =
      detail::PlanCommand(detail::CommandKind::ReusablePrimary);
  constexpr detail::CommandPlan invalid =
      detail::PlanCommand(static_cast<detail::CommandKind>(0xffu));
  return primary.level == VK_COMMAND_BUFFER_LEVEL_PRIMARY &&
         primary.pool_flags ==
             VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT &&
         primary.begin_flags == VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT &&
         primary.fence_flags == 0u && primary.fenced && !primary.inherited &&
         primary.valid &&
         secondary.level == VK_COMMAND_BUFFER_LEVEL_SECONDARY &&
         secondary.pool_flags == 0u && secondary.begin_flags == 0u &&
         secondary.fence_flags == 0u && !secondary.fenced &&
         secondary.inherited && secondary.valid &&
         reusable.level == VK_COMMAND_BUFFER_LEVEL_PRIMARY &&
         reusable.pool_flags == 0u && reusable.begin_flags == 0u &&
         reusable.fence_flags == VK_FENCE_CREATE_SIGNALED_BIT &&
         reusable.fenced && !reusable.inherited && reusable.valid &&
         !invalid.valid;
}

static_assert(VulkanCommandResourceContract());
#endif

[[nodiscard]] bool VulkanCommandFailureContract() {
#if defined(RUND_NODE_HAVE_VULKAN_SDK)
  detail::VulkanCommand command{};
  const rund::AccelCheck rejected = detail::CreateCommand(
      VK_NULL_HANDLE, 0u, command, detail::CommandKind::ReusablePrimary);
  return !rejected.ok &&
         rejected.reason ==
             std::string_view{"accel_vulkan_command_unavailable"} &&
         command.pool == VK_NULL_HANDLE && command.buffer == VK_NULL_HANDLE &&
         command.fence == VK_NULL_HANDLE;
#else
  return true;
#endif
}

template <class... Counter>
void OverflowCounters(Counter &...counter) noexcept {
  ((counter = kCounterMaximum - 1u,
    ::rund::detail::counter::Accumulate(counter, 2u)),
   ...);
}

template <class... Value>
[[nodiscard]] bool CountersEqual(const std::uint64_t expected,
                                 const Value... value) noexcept {
  return ((value == expected) && ...);
}

[[nodiscard]] rund::AccelDevice Pick(const rund::AccelApi api) {
  return rund::node::accel::PickAccel(
      node_accel_contract::backend::Policy({api}));
}

[[nodiscard]] bool PickTokenAdmissionContract() {
  if (detail::AdmitPick(rund::AccelDevice{}) != nullptr) {
    return false;
  }
  rund::AccelDevice pick = Pick(rund::AccelApi::Cpu);
  std::shared_ptr<detail::PickToken> token = detail::AdmitPick(pick);
  if (!pick.check.ok || token == nullptr || token->ops == nullptr ||
      token->raw.api != pick.api || token->raw.backend.context == nullptr) {
    return false;
  }
  const detail::BackendOps *const ops = token->ops;
  token.reset();
  const std::shared_ptr<detail::PickToken> retained = detail::AdmitPick(pick);
  if (retained == nullptr || retained->ops != ops ||
      retained->raw.backend.context == nullptr) {
    return false;
  }
  pick.api = rund::AccelApi::Fake;
  return detail::AdmitPick(pick) == nullptr;
}

template <class Adapter>
[[nodiscard]] bool WaitForActiveHostReadback(Adapter *const adapter) {
  const auto deadline =
      std::chrono::steady_clock::now() + std::chrono::seconds{5};
  while (std::chrono::steady_clock::now() < deadline) {
    {
      std::lock_guard lock{adapter->mutex};
      if (adapter->active_host_readbacks != 0u) {
        return true;
      }
    }
    std::this_thread::yield();
  }
  return false;
}

template <class Adapter>
[[nodiscard]] bool HostReadbackEpochContract(rund::AccelDevice pick,
                                             Adapter *const adapter,
                                             const bool destroy_owner) {
  constexpr std::size_t kBytes = 16u * 1024u * 1024u;
  if (adapter == nullptr) {
    return false;
  }
  std::shared_ptr<detail::PickToken> token = detail::AdmitPick(pick);
  if (token == nullptr) {
    return false;
  }
  rund::Buffer buffer = detail::CreateBackendBuffer(
      token, rund::BufferDesc{.bytes = kBytes,
                              .usage = rund::BufferUsage::ReadWrite,
                              .alignment = 16u});
  std::vector<std::uint8_t> input(kBytes, 0x5au);
  std::vector<std::uint8_t> output(kBytes);
  if (!buffer.check.ok || !detail::UploadBackendBuffer(
                               token, buffer, input.data(), input.size(), 0u)
                               .ok) {
    return false;
  }
  rund::node::accel::ResetRuntimeStats(pick);

  detail::BackendDownload downloaded{};
  std::thread reader{[token, owned_buffer = buffer, &downloaded, &output] {
    downloaded = detail::DownloadBackendBuffer(
        token, owned_buffer, output.data(), output.size(), 0u, true);
  }};
  if (!WaitForActiveHostReadback(adapter)) {
    reader.join();
    return false;
  }

  if (destroy_owner) {
    buffer = {};
    token.reset();
    pick = {};
  } else {
    rund::node::accel::ResetRuntimeStats(pick);
  }
  reader.join();
  if (!downloaded.check.ok || !downloaded.payload_hash_valid ||
      output != input) {
    return false;
  }
  if (destroy_owner) {
    return true;
  }
  const rund::RuntimeStats after = rund::node::accel::ReadRuntimeStats(pick);
  return after.ok && after.device_to_host_bytes == 0u &&
         after.readback_ns == 0u;
}

[[nodiscard]] bool MetalHostReadbackContracts() {
  rund::AccelDevice epoch_pick = Pick(rund::AccelApi::Metal);
  if (!epoch_pick.check.ok) {
    return node_accel_contract::MetalFailsClosed(epoch_pick);
  }
  const std::shared_ptr<detail::PickToken> epoch_token =
      detail::AdmitPick(epoch_pick);
  const rund::AccelDevice *const epoch_raw =
      epoch_token == nullptr ? nullptr : &epoch_token->raw;
  auto *const epoch_adapter =
      epoch_raw == nullptr
          ? nullptr
          : static_cast<detail::MetalAdapter *>(epoch_raw->backend.context);
  if (!HostReadbackEpochContract(epoch_pick, epoch_adapter, false)) {
    return false;
  }

  rund::AccelDevice destroy_pick = Pick(rund::AccelApi::Metal);
  std::shared_ptr<detail::PickToken> destroy_token =
      detail::AdmitPick(destroy_pick);
  const rund::AccelDevice *const destroy_raw =
      destroy_token == nullptr ? nullptr : &destroy_token->raw;
  auto *const destroy_adapter =
      destroy_raw == nullptr
          ? nullptr
          : static_cast<detail::MetalAdapter *>(destroy_raw->backend.context);
  destroy_token.reset();
  return HostReadbackEpochContract(std::move(destroy_pick), destroy_adapter,
                                   true);
}

[[nodiscard]] bool VulkanHostReadbackContracts() {
  rund::AccelDevice epoch_pick = Pick(rund::AccelApi::Vulkan);
  if (!epoch_pick.check.ok) {
    return node_accel_contract::vulkan::FailureReasonIsPrecise(epoch_pick);
  }
#if defined(RUND_NODE_HAVE_VULKAN_SDK)
  const std::shared_ptr<detail::PickToken> epoch_token =
      detail::AdmitPick(epoch_pick);
  const rund::AccelDevice *const epoch_raw =
      epoch_token == nullptr ? nullptr : &epoch_token->raw;
  detail::VulkanAdapter *const epoch_adapter =
      epoch_raw == nullptr ? nullptr : detail::CheckedVulkanAdapter(*epoch_raw);
  if (!HostReadbackEpochContract(epoch_pick, epoch_adapter, false)) {
    return false;
  }

  rund::AccelDevice destroy_pick = Pick(rund::AccelApi::Vulkan);
  std::shared_ptr<detail::PickToken> destroy_token =
      detail::AdmitPick(destroy_pick);
  const rund::AccelDevice *const destroy_raw =
      destroy_token == nullptr ? nullptr : &destroy_token->raw;
  detail::VulkanAdapter *const destroy_adapter =
      destroy_raw == nullptr ? nullptr
                             : detail::CheckedVulkanAdapter(*destroy_raw);
  destroy_token.reset();
  return HostReadbackEpochContract(std::move(destroy_pick), destroy_adapter,
                                   true);
#else
  return false;
#endif
}

[[nodiscard]] bool CpuCounterContract(const rund::AccelDevice &pick) {
  const std::shared_ptr<detail::PickToken> token = detail::AdmitPick(pick);
  const rund::AccelDevice *const raw = token == nullptr ? nullptr : &token->raw;
  detail::CpuAdapter *const adapter =
      raw == nullptr ? nullptr : detail::CpuAdapterFromPick(*raw);
  if (adapter == nullptr) {
    return false;
  }
  {
    std::lock_guard lock{adapter->mutex};
    OverflowCounters(adapter->dispatch_count, adapter->buffer_allocation_count,
                     adapter->host_to_device_bytes,
                     adapter->device_to_host_bytes);
  }
  const rund::RuntimeStats saturated =
      rund::node::accel::ReadRuntimeStats(pick);
  const bool saturated_ok =
      saturated.ok && CountersEqual(kCounterMaximum, saturated.dispatch_count,
                                    saturated.buffer_allocation_count,
                                    saturated.host_to_device_bytes,
                                    saturated.device_to_host_bytes);
  rund::node::accel::ResetRuntimeStats(pick);
  const rund::RuntimeStats reset = rund::node::accel::ReadRuntimeStats(pick);
  return saturated_ok && reset.ok &&
         CountersEqual(0u, reset.dispatch_count, reset.buffer_allocation_count,
                       reset.host_to_device_bytes, reset.device_to_host_bytes);
}

[[nodiscard]] bool MetalCounterContract(const rund::AccelDevice &pick) {
  if (!pick.check.ok) {
    return node_accel_contract::MetalFailsClosed(pick);
  }
  const std::shared_ptr<detail::PickToken> token = detail::AdmitPick(pick);
  const rund::AccelDevice *const raw = token == nullptr ? nullptr : &token->raw;
  auto *const adapter =
      raw == nullptr
          ? nullptr
          : static_cast<detail::MetalAdapter *>(raw->backend.context);
  if (adapter == nullptr) {
    return false;
  }
  {
    std::lock_guard lock{adapter->mutex};
    detail::MetalRuntimeStats &stats = adapter->stats;
    OverflowCounters(
        stats.dispatch_count, stats.command_submit_count,
        stats.pipeline_compile_count, stats.pipeline_cache_hit_count,
        stats.buffer_allocation_count, stats.buffer_reuse_hit_count,
        stats.host_to_device_bytes, stats.device_to_host_bytes,
        stats.accel_kernel_ns, stats.accel_timestamp_count,
        stats.shader_compile_ns, stats.spirv_compile_ns,
        stats.pipeline_create_ns, stats.descriptor_setup_ns,
        stats.command_submit_wait_ns, stats.readback_ns);
  }
  const rund::RuntimeStats saturated =
      rund::node::accel::ReadRuntimeStats(pick);
  const bool saturated_ok =
      saturated.ok &&
      CountersEqual(
          kCounterMaximum, saturated.dispatch_count,
          saturated.command_submit_count, saturated.pipeline_compile_count,
          saturated.pipeline_cache_hit_count, saturated.buffer_allocation_count,
          saturated.buffer_reuse_hit_count, saturated.host_to_device_bytes,
          saturated.device_to_host_bytes, saturated.accel_kernel_ns,
          saturated.accel_timestamp_count, saturated.shader_compile_ns,
          saturated.spirv_compile_ns, saturated.pipeline_create_ns,
          saturated.descriptor_setup_ns, saturated.command_submit_wait_ns,
          saturated.readback_ns);
  rund::node::accel::ResetRuntimeStats(pick);
  const rund::RuntimeStats reset = rund::node::accel::ReadRuntimeStats(pick);
  return saturated_ok && reset.ok &&
         CountersEqual(
             0u, reset.dispatch_count, reset.command_submit_count,
             reset.pipeline_compile_count, reset.pipeline_cache_hit_count,
             reset.buffer_allocation_count, reset.buffer_reuse_hit_count,
             reset.host_to_device_bytes, reset.device_to_host_bytes,
             reset.accel_kernel_ns, reset.accel_timestamp_count,
             reset.shader_compile_ns, reset.spirv_compile_ns,
             reset.pipeline_create_ns, reset.descriptor_setup_ns,
             reset.command_submit_wait_ns, reset.readback_ns);
}

[[nodiscard]] bool VulkanCounterContract(const rund::AccelDevice &pick) {
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
    std::lock_guard lock{adapter->mutex};
    OverflowCounters(
        adapter->dispatch_count, adapter->command_submit_count,
        adapter->command_capacity_rejection_count,
        adapter->pipeline_compile_count, adapter->pipeline_cache_hit_count,
        adapter->descriptor_pool_create_count,
        adapter->descriptor_set_allocate_count,
        adapter->descriptor_reuse_hit_count, adapter->buffer_allocation_count,
        adapter->buffer_reuse_hit_count, adapter->host_to_device_bytes,
        adapter->device_to_host_bytes, adapter->accel_kernel_ns,
        adapter->accel_timestamp_count, adapter->shader_compile_ns,
        adapter->spirv_compile_ns, adapter->pipeline_create_ns,
        adapter->descriptor_setup_ns, adapter->command_submit_wait_ns,
        adapter->readback_ns);
    adapter->command_inflight_peak = detail::kVulkanCommandCapacity;
  }
  const rund::RuntimeStats saturated =
      rund::node::accel::ReadRuntimeStats(pick);
  const bool saturated_ok =
      saturated.ok &&
      CountersEqual(
          kCounterMaximum, saturated.dispatch_count,
          saturated.command_submit_count,
          saturated.command_capacity_rejection_count,
          saturated.pipeline_compile_count, saturated.pipeline_cache_hit_count,
          saturated.descriptor_pool_create_count,
          saturated.descriptor_set_allocate_count,
          saturated.descriptor_reuse_hit_count,
          saturated.buffer_allocation_count, saturated.buffer_reuse_hit_count,
          saturated.host_to_device_bytes, saturated.device_to_host_bytes,
          saturated.accel_kernel_ns, saturated.accel_timestamp_count,
          saturated.shader_compile_ns, saturated.spirv_compile_ns,
          saturated.pipeline_create_ns, saturated.descriptor_setup_ns,
          saturated.command_submit_wait_ns, saturated.readback_ns) &&
      saturated.command_capacity == detail::kVulkanCommandCapacity &&
      saturated.command_inflight_peak == detail::kVulkanCommandCapacity;
  rund::node::accel::ResetRuntimeStats(pick);
  const rund::RuntimeStats reset = rund::node::accel::ReadRuntimeStats(pick);
  return saturated_ok && reset.ok &&
         reset.command_capacity == detail::kVulkanCommandCapacity &&
         reset.command_inflight_peak == 0u &&
         reset.command_capacity_rejection_count == 0u &&
         CountersEqual(
             0u, reset.dispatch_count, reset.command_submit_count,
             reset.pipeline_compile_count, reset.pipeline_cache_hit_count,
             reset.descriptor_pool_create_count,
             reset.descriptor_set_allocate_count,
             reset.descriptor_reuse_hit_count, reset.buffer_allocation_count,
             reset.buffer_reuse_hit_count, reset.host_to_device_bytes,
             reset.device_to_host_bytes, reset.accel_kernel_ns,
             reset.accel_timestamp_count, reset.shader_compile_ns,
             reset.spirv_compile_ns, reset.pipeline_create_ns,
             reset.descriptor_setup_ns, reset.command_submit_wait_ns,
             reset.readback_ns);
#else
  return false;
#endif
}

[[nodiscard]] bool VulkanMemoryTierContract(const rund::AccelDevice &pick) {
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
    if (!stats.ok || stats.host_to_device_bytes != 10u ||
        stats.device_to_host_bytes != 7u || stats.command_submit_count < 4u) {
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

[[nodiscard]] bool MetalRuntimeContracts(const rund::AccelDevice &pick) {
  if (!pick.check.ok) {
    return node_accel_contract::MetalFailsClosed(pick);
  }
  TEST_ASSERT(
      node_accel_contract::MetalRepeatedStagedRunsReportWarmRuntimeStats(pick));
  TEST_ASSERT(
      node_accel_contract::MetalResidentBufferRegistryValidatesPrivateRefs(
          pick));
  TEST_ASSERT(node_accel_contract::PublicBufferApiContract(pick));
  return true;
}

[[nodiscard]] bool VulkanRuntimeContracts() {
  TEST_ASSERT(
      node_accel_contract::VulkanRepeatedStagedRunsReportWarmRuntimeStats());
  return true;
}

} // namespace

int RunAccelBackendRuntimeContract() {
  TEST_ASSERT(PickTokenAdmissionContract());
  TEST_ASSERT(node_accel_contract::ResidentValidationContract());
  TEST_ASSERT(VulkanCommandFailureContract());
  TEST_ASSERT(CpuCounterContract(Pick(rund::AccelApi::Cpu)));
  TEST_ASSERT(MetalCounterContract(Pick(rund::AccelApi::Metal)));
  TEST_ASSERT(VulkanCounterContract(Pick(rund::AccelApi::Vulkan)));
  TEST_ASSERT(MetalHostReadbackContracts());
  TEST_ASSERT(VulkanHostReadbackContracts());
  TEST_ASSERT(VulkanMemoryTierContract(Pick(rund::AccelApi::Vulkan)));
  TEST_ASSERT(MetalRuntimeContracts(Pick(rund::AccelApi::Metal)));
  TEST_ASSERT(VulkanRuntimeContracts());
  return 0;
}

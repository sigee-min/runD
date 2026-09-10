#pragma once

#include <accel/check.hpp>
#include <accel/device.hpp>

#include "../kernel/backend/run.hpp"
#include "../kernel/callback.hpp"
#include "../kernel/memory.hpp"
#include "../kernel/preparation.hpp"
#include "../kernel/residency/device_vsm.hpp"

#include <cstddef>
#include <cstdint>
#include <memory>

namespace rund::node::accel::detail {

struct VulkanAdapter;

using VulkanDeviceVsmPreparation = DeviceVsmPreparation;
[[nodiscard]] VulkanDeviceVsmPreparation
PrepareVulkanDeviceVsm(const rund::AccelDevice &,
                       const std::shared_ptr<const DeviceVsmProof> &) noexcept;

[[nodiscard]] rund::AccelCheck PrepareVulkanResources(
    const rund::AccelDevice &pick, const BoundStep *steps,
    std::size_t step_count, std::uint64_t dispatch_count,
    KernelPreparationMode mode, const BoundResets *resets,
    const KernelViewLayout *views, const RunBinds *view_binds,
    const KernelScratchLayout *scratch, const BackendRun *template_probe,
    PreparedKernelTemplateRegistry *templates, std::uint32_t *failed_node,
    std::shared_ptr<void> &prepared, PreparedMemory &memory);
[[nodiscard]] std::uint64_t
VulkanKernelTraffic(const std::shared_ptr<void> &prepared) noexcept;
[[nodiscard]] PreparedMemory
ObserveVulkanPipelineTemplate(const void *prepared) noexcept;
[[nodiscard]] rund::AccelCheck
RunVulkanResources(const rund::AccelDevice &pick,
                   const std::shared_ptr<void> &prepared);
[[nodiscard]] rund::AccelCheck SubmitVulkanResources(
    const rund::AccelDevice &pick, const std::shared_ptr<void> &prepared,
    KernelCompletion completion, void *user, PreparedMemoryMeter *memory,
    KernelTiming timing) noexcept;
[[nodiscard]] rund::AccelCheck
SeedPreparedVulkanPipelineGeneration(const std::shared_ptr<void> &prepared,
                                     std::uint32_t generation) noexcept;

// Exact Vulkan-private shape of the already admitted fused Direct Map
// recurrence. `iterations` is carried by the fixed eight-byte push-constant
// row and consumed by one device-side shader loop. Vulkan does not expose the
// implementation's VkCommandBuffer allocation bytes, so fixed native storage
// is proved by fixed object/descriptor counts plus exact runD-owned buffers.
enum class VulkanFusedDirectRecurrenceRetention : std::uint8_t {
  Unknown,
  Terminal,
  History,
};
struct VulkanFusedDirectRecurrenceDiagnostics final {
  std::uint64_t iteration_count{};
  std::uint64_t native_command_buffer_count{};
  std::uint64_t native_dispatch_count{};
  std::uint64_t descriptor_set_count{};
  std::uint64_t device_buffer_bytes{};
  std::uint64_t route_host_bytes{};
  std::uint64_t push_constant_bytes{};
  bool command_buffer_allocation_bytes_observable{};
  VulkanFusedDirectRecurrenceRetention retention{
      VulkanFusedDirectRecurrenceRetention::Unknown};
};
[[nodiscard]] bool InspectVulkanFusedDirectRecurrence(
    const std::shared_ptr<void> &,
    VulkanFusedDirectRecurrenceDiagnostics &) noexcept;
} // namespace rund::node::accel::detail

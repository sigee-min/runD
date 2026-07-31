#pragma once

#include <accel/api.hpp>
#include <accel/buffer.hpp>
#include <accel/context/buffer/descriptor.hpp>
#include <accel/device.hpp>
#include <accel/kernel/evidence.hpp>
#include <accel/kernel/value.hpp>

#include <array>
#include <cstddef>
#include <cstdint>
#include <kernel/program/compute/dsl.hpp>
#include <string_view>

namespace node_accel_contract::cpu_context {

struct MapRun {
  std::uint64_t hash = 0u;
  rund::AccelEvidence evidence{};
  bool ok = false;
};

[[nodiscard]] constexpr rund::AccelPolicy CpuPolicy() noexcept {
  return rund::AccelPolicy{
      .preferred = {rund::AccelApi::Cpu, rund::AccelApi::Auto,
                    rund::AccelApi::Auto},
      .preferred_count = 1u,
      .allow_fake = false,
  };
}

[[nodiscard]] constexpr rund::AccelPolicy
ApiPolicy(const rund::AccelApi api) noexcept {
  return rund::AccelPolicy{
      .preferred = {api, rund::AccelApi::Auto, rund::AccelApi::Auto},
      .preferred_count = 1u,
      .allow_fake = false,
  };
}

[[nodiscard]] constexpr rund::AccelBufferDesc
BufferDesc(const rund::BufferUsage usage, const std::uint64_t count) noexcept {
  return rund::AccelBufferDesc{
      .scalar_width_bytes = sizeof(rund::kernel::i32),
      .count = count,
      .usage = usage,
  };
}

template <typename T>
[[nodiscard]] std::uint64_t HashValues(const T *const values,
                                       const std::size_t count) noexcept {
  std::uint64_t hash = 1469598103934665603ull;
  const auto *const bytes = reinterpret_cast<const std::uint8_t *>(values);
  for (std::size_t index = 0u; index < count * sizeof(T); ++index) {
    hash ^= bytes[index];
    hash *= 1099511628211ull;
  }
  return hash;
}

[[nodiscard]] inline bool
EvidenceReason(const rund::AccelEvidence &evidence,
               const std::string_view reason) noexcept {
  return !evidence.ok && std::string_view{evidence.reason} == reason;
}

[[nodiscard]] inline bool KernelReason(const rund::AccelKernel &kernel,
                                       const std::string_view reason) noexcept {
  return !kernel.check.ok && std::string_view{kernel.check.reason} == reason;
}

[[nodiscard]] inline bool
PickUnavailableReasonIsPrecise(const rund::AccelDevice &pick,
                               const rund::AccelApi api) noexcept {
  if (pick.check.ok) {
    return false;
  }
  const std::string_view reason{pick.check.reason};
  if (api == rund::AccelApi::Metal) {
    return reason == "accel_metal_unavailable" ||
           reason == "accel_metal_device_unavailable" ||
           reason == "accel_metal_queue_unavailable" ||
           reason == "accel_metal_sdk_unavailable";
  }
  return reason == "accel_vulkan_loader_unavailable" ||
         reason == "accel_vulkan_instance_unavailable" ||
         reason == "accel_vulkan_portability_unavailable" ||
         reason == "accel_vulkan_device_unavailable" ||
         reason == "accel_vulkan_queue_unavailable" ||
         reason == "accel_vulkan_shader_tool_unavailable" ||
         reason == "accel_vulkan_unavailable";
}

[[nodiscard]] MapRun ContextMapHash(const rund::AccelDevice &pick);
[[nodiscard]] bool CpuContextRunsMap(const rund::AccelDevice &pick);
[[nodiscard]] bool
CpuContextMatchesAvailableRealBackend(const rund::AccelDevice &cpu_pick);
[[nodiscard]] bool CpuContextRunsScanThenMap(const rund::AccelDevice &pick);
[[nodiscard]] bool
CpuContextRunsSegmentedScanThenMap(const rund::AccelDevice &pick);
[[nodiscard]] bool CpuContextRunsSegmentedReduce(const rund::AccelDevice &pick);
[[nodiscard]] bool
CpuContextRunsSegmentedReduceRangeDefault(const rund::AccelDevice &pick);
[[nodiscard]] bool CpuContextRejectsBadScanHash(const rund::AccelDevice &pick);
[[nodiscard]] bool
CpuContextRejectsForeignBuffer(const rund::AccelDevice &pick);

} // namespace node_accel_contract::cpu_context

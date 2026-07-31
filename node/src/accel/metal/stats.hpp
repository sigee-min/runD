#pragma once

#include <accel/device.hpp>


#include <cstdint>

namespace rund::node::accel::detail {

struct MetalRuntimeStats {
  std::uint64_t dispatch_count = 0u;
  std::uint64_t command_submit_count = 0u;
  std::uint64_t pipeline_compile_count = 0u;
  std::uint64_t pipeline_cache_hit_count = 0u;
  std::uint64_t library_compile_count = 0u;
  std::uint64_t library_cache_hit_count = 0u;
  std::uint64_t buffer_allocation_count = 0u;
  std::uint64_t buffer_reuse_hit_count = 0u;
  std::uint64_t host_to_device_bytes = 0u;
  std::uint64_t device_to_host_bytes = 0u;
  std::uint64_t accel_kernel_ns = 0u;
  std::uint64_t accel_timestamp_count = 0u;
  const char* accel_timestamp_source = "unavailable";
  std::uint64_t shader_compile_ns = 0u;
  std::uint64_t spirv_compile_ns = 0u;
  std::uint64_t pipeline_create_ns = 0u;
  std::uint64_t descriptor_setup_ns = 0u;
  std::uint64_t command_submit_wait_ns = 0u;
  std::uint64_t readback_ns = 0u;
  bool ok = false;
  const char* reason = "accel_runtime_stats_invalid";
};

struct MetalMemoryStats final {
  std::uint64_t current{};
  std::uint64_t peak{};
  std::uint64_t cumulative{};
  std::uint64_t reused{};
};

[[nodiscard]] MetalRuntimeStats ReadMetalRuntimeStats(const rund::AccelDevice& pick);
[[nodiscard]] MetalMemoryStats ReadMetalMemoryStats(
    const rund::AccelDevice& pick) noexcept;
void ResetMetalRuntimeStats(const rund::AccelDevice& pick);

}  // namespace rund::node::accel::detail

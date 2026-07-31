#pragma once

#include <cstdint>

namespace rund {

struct RuntimeStats {
  std::uint64_t dispatch_count = 0u;
  std::uint64_t command_submit_count = 0u;
  std::uint64_t command_capacity = 0u;
  std::uint64_t command_inflight_peak = 0u;
  std::uint64_t command_capacity_rejection_count = 0u;
  std::uint64_t reset_command_count = 0u;
  std::uint64_t reset_bytes = 0u;
  std::uint64_t pipeline_compile_count = 0u;
  std::uint64_t pipeline_cache_hit_count = 0u;
  std::uint64_t pipeline_cache_eviction_count = 0u;
  std::uint64_t descriptor_pool_create_count = 0u;
  std::uint64_t descriptor_set_allocate_count = 0u;
  std::uint64_t descriptor_reuse_hit_count = 0u;
  std::uint64_t buffer_allocation_count = 0u;
  std::uint64_t buffer_reuse_hit_count = 0u;
  std::uint64_t host_to_device_bytes = 0u;
  std::uint64_t device_to_host_bytes = 0u;
  std::uint64_t accel_kernel_ns = 0u;
  std::uint64_t accel_timestamp_count = 0u;
  const char *accel_timestamp_source = "unavailable";
  std::uint64_t shader_compile_ns = 0u;
  std::uint64_t spirv_compile_ns = 0u;
  std::uint64_t pipeline_create_ns = 0u;
  std::uint64_t descriptor_setup_ns = 0u;
  std::uint64_t command_submit_wait_ns = 0u;
  std::uint64_t readback_ns = 0u;
  std::uint64_t generated_item_count = 0u;
  std::uint64_t generated_capacity = 0u;
  std::uint64_t indirect_dispatch_count = 0u;
  std::uint64_t indirect_work_item_count = 0u;
  std::uint64_t iteration_count = 0u;
  std::uint64_t skipped_iteration_count = 0u;
  std::uint64_t conflict_count = 0u;
  std::uint64_t overflow_ordinal = ~std::uint64_t{0u};
  bool ok = false;
  const char *reason = "accel_runtime_stats_invalid";
};

} // namespace rund

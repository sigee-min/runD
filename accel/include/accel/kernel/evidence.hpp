#pragma once

#include <accel/api.hpp>

#include <cstdint>

namespace rund {

struct AccelEvidence {
  std::uint64_t graph_id_hi = 0u;
  std::uint64_t graph_id_lo = 0u;
  std::uint64_t kernel_id = 0u;
  AccelApi backend = AccelApi::Auto;
  std::uint64_t dispatch_count = 0u;
  std::uint64_t command_submit_count = 0u;
  std::uint64_t command_capacity = 0u;
  std::uint64_t command_inflight_peak = 0u;
  std::uint64_t command_capacity_rejection_count = 0u;
  std::uint64_t reset_command_count = 0u;
  std::uint64_t reset_bytes = 0u;
  std::uint64_t original_operation_count = 0u;
  std::uint64_t fused_operation_count = 0u;
  std::uint64_t original_dispatch_count = 0u;
  std::uint64_t final_dispatch_count = 0u;
  std::uint64_t fusion_rejection_count = 0u;
  const char *fusion_reason = "compute_fusion_invalid";
  std::uint64_t internal_producer_consumer_roundtrip_bytes = 0u;
  std::uint64_t external_producer_consumer_roundtrip_bytes = 0u;
  std::uint64_t host_to_device_bytes = 0u;
  std::uint64_t device_to_host_bytes = 0u;
  std::uint64_t pipeline_compile_count = 0u;
  std::uint64_t pipeline_cache_hit_count = 0u;
  std::uint64_t pipeline_cache_eviction_count = 0u;
  std::uint64_t descriptor_pool_create_count = 0u;
  std::uint64_t descriptor_set_allocate_count = 0u;
  std::uint64_t buffer_reuse_hit_count = 0u;
  std::uint64_t buffer_allocation_count = 0u;
  std::uint64_t descriptor_reuse_hit_count = 0u;
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
  std::uint64_t failed_batches = 0u;
  std::uint64_t first_failed_batch = 0u;
  std::uint32_t first_status = 0u;
  bool ok = false;
  const char *reason = "accel_kernel_run_invalid";
};

} // namespace rund

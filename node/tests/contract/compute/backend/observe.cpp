#include "local.hpp"

#include "../../../../src/compute/stats.hpp"

#include <accel/kernel/evidence.hpp>

#include <bit>
#include <cstdint>

namespace rund_node_backend_contract {

[[nodiscard]] bool CheckStatsProjection() {
  const rund::AccelEvidence evidence{
      .identity = {.graph_id_hi = 0x123456789abcdef0ull,
                   .graph_id_lo = 0x0fedcba987654321ull,
                   .kernel_id = 73u,
                   .backend = rund::AccelApi::Vulkan},
      .run = {.work = {.dispatch_count = 3u,
                       .command_submit_count = 1u,
                       .command_capacity = 8u,
                       .command_inflight_peak = 5u,
                       .command_capacity_rejection_count = 2u,
                       .reset_command_count = 7u,
                       .reset_bytes = 97u,
                       .original_operation_count = 11u,
                       .fused_operation_count = 7u,
                       .original_dispatch_count = 2u,
                       .final_dispatch_count = 3u,
                       .fusion_rejection_count = 5u,
                       .fusion_reason = "projected",
                       .generated_item_count = 79u,
                       .generated_capacity = 83u,
                       .indirect_dispatch_count = 89u,
                       .indirect_work_item_count = 97u,
                       .iteration_count = 101u,
                       .skipped_iteration_count = 103u,
                       .conflict_count = 107u,
                       .overflow_ordinal = 109u},
              .time = {.accel_kernel_ns = 43u,
                       .accel_timestamp_count = 2u,
                       .accel_timestamp_source = "test_clock",
                       .shader_compile_ns = 47u,
                       .spirv_compile_ns = 53u,
                       .pipeline_create_ns = 59u,
                       .descriptor_setup_ns = 61u,
                       .command_submit_wait_ns = 67u,
                       .readback_ns = 71u},
              .transfer = {.host_to_device_bytes = 107u,
                           .device_to_host_bytes = 109u,
                           .internal_producer_consumer_roundtrip_bytes = 101u,
                           .external_producer_consumer_roundtrip_bytes = 103u},
              .allocations = {.pipeline_compile_count = 13u,
                              .pipeline_cache_hit_count = 17u,
                              .pipeline_cache_eviction_count = 19u,
                              .descriptor_pool_create_count = 23u,
                              .descriptor_set_allocate_count = 29u,
                              .descriptor_reuse_hit_count = 41u,
                              .buffer_allocation_count = 37u,
                              .buffer_reuse_hit_count = 31u}},
      .outcome = {.ok = true, .reason = "ok"},
  };
  const rund::compute::Stats stats = rund::compute::detail::stats_from_evidence(
      rund::compute::Backend::Vulkan, evidence, 113u);
  const std::uint64_t graph = evidence.identity.graph_id_hi ^
                              std::rotl(evidence.identity.graph_id_lo, 1);
  return stats.backend == rund::compute::Backend::Vulkan &&
         stats.pipeline_compiles == 13u && stats.buffer_allocations == 37u &&
         stats.download_events == 0u && stats.dispatches == 3u &&
         stats.command_submits == 1u && stats.command_capacity == 8u &&
         stats.transfer_submissions.host_to_device == 0u &&
         stats.transfer_submissions.device_to_host == 0u &&
         stats.transfer_submissions.device_to_device == 0u &&
         stats.command_inflight_peak == 5u &&
         stats.command_capacity_rejections == 2u &&
         stats.reset_commands == 7u && stats.reset_bytes == 97u &&
         stats.dispatches != stats.command_submits &&
         stats.uploaded_bytes == 107u && stats.downloaded_bytes == 109u &&
         stats.pipeline_cache_hits == 17u &&
         stats.pipeline_cache_evictions == 19u && stats.buffer_reuses == 31u &&
         stats.descriptor_pool_creations == 23u &&
         stats.descriptor_set_allocations == 29u &&
         stats.descriptor_reuses == 41u && stats.original_dispatches == 2u &&
         stats.final_dispatches == 3u && stats.fusions == 4u &&
         stats.fusion_rejections == 5u &&
         stats.internal_roundtrip_bytes == 101u &&
         stats.external_roundtrip_bytes == 103u && stats.kernel_ns == 43u &&
         stats.graph_read_bytes == 113u && stats.kernel_samples == 2u &&
         stats.kernel_timing_available() && stats.shader_compile_ns == 47u &&
         stats.spirv_compile_ns == 53u && stats.pipeline_create_ns == 59u &&
         stats.descriptor_setup_ns == 61u && stats.submit_wait_ns == 67u &&
         stats.readback_ns == 71u && stats.graph_hash == graph &&
         stats.output_hash == 0u && stats.control.generated_item_count == 79u &&
         stats.control.generated_capacity == 83u &&
         stats.control.indirect_dispatch_count == 89u &&
         stats.control.indirect_work_item_count == 97u &&
         stats.control.iteration_count == 101u &&
         stats.control.skipped_iteration_count == 103u &&
         stats.control.conflict_count == 107u &&
         stats.control.overflow_ordinal == 109u &&
         !rund::compute::Stats{}.kernel_timing_available();
}

} // namespace rund_node_backend_contract

#include <accel/context/value.hpp>
#include <accel/kernel/evidence.hpp>
#include <accel/runtime.hpp>

#include "evidence.hpp"

namespace rund::node::accel::detail {

rund::AccelEvidence RejectKernelEvidence(const rund::AccelContext &context,
                                         const KernelExecution &execution,
                                         const char *const reason) {
  if (!execution.admission.check.ok) {
    return rund::AccelEvidence{.ok = false, .reason = reason};
  }
  return rund::AccelEvidence{
      .graph_id_hi = execution.admission.graph_id_hi,
      .graph_id_lo = execution.admission.graph_id_lo,
      .kernel_id = execution.admission.kernel_id,
      .backend = context.api,
      .original_operation_count = execution.original_operation_count,
      .fused_operation_count = execution.fused_operation_count,
      .fusion_rejection_count = execution.fusion_rejection_count,
      .fusion_reason = execution.fusion_reason,
      .ok = false,
      .reason = reason,
  };
}

rund::AccelEvidence EvidenceFromStats(
    const rund::AccelContext &context, const KernelExecution &execution,
    const rund::RuntimeStats &stats,
    const std::uint64_t original_dispatch_count,
    const std::uint64_t final_dispatch_count, const bool ok,
    const char *const reason,
    const std::uint64_t internal_producer_consumer_roundtrip_bytes,
    const std::uint64_t external_producer_consumer_roundtrip_bytes,
    const std::uint64_t failed_batches, const std::uint64_t first_failed_batch,
    const std::uint32_t first_status) {
  return rund::AccelEvidence{
      .graph_id_hi = execution.admission.graph_id_hi,
      .graph_id_lo = execution.admission.graph_id_lo,
      .kernel_id = execution.admission.kernel_id,
      .backend = context.api,
      .dispatch_count = stats.dispatch_count,
      .command_submit_count = stats.command_submit_count,
      .command_capacity = stats.command_capacity,
      .command_inflight_peak = stats.command_inflight_peak,
      .command_capacity_rejection_count =
          stats.command_capacity_rejection_count,
      .reset_command_count = stats.reset_command_count,
      .reset_bytes = stats.reset_bytes,
      .original_operation_count = execution.original_operation_count,
      .fused_operation_count = execution.fused_operation_count,
      .original_dispatch_count = original_dispatch_count,
      .final_dispatch_count = final_dispatch_count,
      .fusion_rejection_count = execution.fusion_rejection_count,
      .fusion_reason = execution.fusion_reason,
      .internal_producer_consumer_roundtrip_bytes =
          internal_producer_consumer_roundtrip_bytes,
      .external_producer_consumer_roundtrip_bytes =
          external_producer_consumer_roundtrip_bytes,
      .host_to_device_bytes = stats.host_to_device_bytes,
      .device_to_host_bytes = stats.device_to_host_bytes,
      .pipeline_compile_count = stats.pipeline_compile_count,
      .pipeline_cache_hit_count = stats.pipeline_cache_hit_count,
      .pipeline_cache_eviction_count = stats.pipeline_cache_eviction_count,
      .descriptor_pool_create_count = stats.descriptor_pool_create_count,
      .descriptor_set_allocate_count = stats.descriptor_set_allocate_count,
      .buffer_reuse_hit_count = stats.buffer_reuse_hit_count,
      .buffer_allocation_count = stats.buffer_allocation_count,
      .descriptor_reuse_hit_count = stats.descriptor_reuse_hit_count,
      .accel_kernel_ns = stats.accel_kernel_ns,
      .accel_timestamp_count = stats.accel_timestamp_count,
      .accel_timestamp_source = stats.accel_timestamp_source,
      .shader_compile_ns = stats.shader_compile_ns,
      .spirv_compile_ns = stats.spirv_compile_ns,
      .pipeline_create_ns = stats.pipeline_create_ns,
      .descriptor_setup_ns = stats.descriptor_setup_ns,
      .command_submit_wait_ns = stats.command_submit_wait_ns,
      .readback_ns = stats.readback_ns,
      .generated_item_count = stats.generated_item_count,
      .generated_capacity = stats.generated_capacity,
      .indirect_dispatch_count = stats.indirect_dispatch_count,
      .indirect_work_item_count = stats.indirect_work_item_count,
      .iteration_count = stats.iteration_count,
      .skipped_iteration_count = stats.skipped_iteration_count,
      .conflict_count = stats.conflict_count,
      .overflow_ordinal = stats.overflow_ordinal,
      .failed_batches = failed_batches,
      .first_failed_batch = first_failed_batch,
      .first_status = first_status,
      .ok = ok,
      .reason = reason,
  };
}

} // namespace rund::node::accel::detail

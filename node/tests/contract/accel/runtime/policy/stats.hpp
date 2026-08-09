#pragma once

#include <accel/runtime.hpp>

#include "local.hpp"

namespace node_accel_contract::policy_case {

[[nodiscard]] inline bool RuntimeStatsEqual(const rund::RuntimeStats &lhs,
                                            const rund::RuntimeStats &rhs) {
  const auto &left_work = lhs.run.work;
  const auto &right_work = rhs.run.work;
  const auto &left_time = lhs.run.time;
  const auto &right_time = rhs.run.time;
  const auto &left_transfer = lhs.run.transfer;
  const auto &right_transfer = rhs.run.transfer;
  const auto &left_allocations = lhs.run.allocations;
  const auto &right_allocations = rhs.run.allocations;
  return left_work.dispatch_count == right_work.dispatch_count &&
         left_work.command_submit_count == right_work.command_submit_count &&
         left_work.command_capacity == right_work.command_capacity &&
         left_work.command_inflight_peak == right_work.command_inflight_peak &&
         left_work.command_capacity_rejection_count ==
             right_work.command_capacity_rejection_count &&
         left_work.reset_command_count == right_work.reset_command_count &&
         left_work.reset_bytes == right_work.reset_bytes &&
         left_work.original_operation_count ==
             right_work.original_operation_count &&
         left_work.fused_operation_count == right_work.fused_operation_count &&
         left_work.original_dispatch_count ==
             right_work.original_dispatch_count &&
         left_work.final_dispatch_count == right_work.final_dispatch_count &&
         left_work.fusion_rejection_count ==
             right_work.fusion_rejection_count &&
         std::string_view{left_work.fusion_reason} ==
             right_work.fusion_reason &&
         left_work.generated_item_count == right_work.generated_item_count &&
         left_work.generated_capacity == right_work.generated_capacity &&
         left_work.indirect_dispatch_count ==
             right_work.indirect_dispatch_count &&
         left_work.indirect_work_item_count ==
             right_work.indirect_work_item_count &&
         left_work.iteration_count == right_work.iteration_count &&
         left_work.skipped_iteration_count ==
             right_work.skipped_iteration_count &&
         left_work.conflict_count == right_work.conflict_count &&
         left_work.overflow_ordinal == right_work.overflow_ordinal &&
         left_time.accel_kernel_ns == right_time.accel_kernel_ns &&
         left_time.accel_timestamp_count == right_time.accel_timestamp_count &&
         std::string_view{left_time.accel_timestamp_source} ==
             right_time.accel_timestamp_source &&
         left_time.shader_compile_ns == right_time.shader_compile_ns &&
         left_time.spirv_compile_ns == right_time.spirv_compile_ns &&
         left_time.pipeline_create_ns == right_time.pipeline_create_ns &&
         left_time.descriptor_setup_ns == right_time.descriptor_setup_ns &&
         left_time.command_submit_wait_ns ==
             right_time.command_submit_wait_ns &&
         left_time.readback_ns == right_time.readback_ns &&
         left_transfer.host_to_device_bytes ==
             right_transfer.host_to_device_bytes &&
         left_transfer.device_to_host_bytes ==
             right_transfer.device_to_host_bytes &&
         left_transfer.internal_producer_consumer_roundtrip_bytes ==
             right_transfer.internal_producer_consumer_roundtrip_bytes &&
         left_transfer.external_producer_consumer_roundtrip_bytes ==
             right_transfer.external_producer_consumer_roundtrip_bytes &&
         left_allocations.pipeline_compile_count ==
             right_allocations.pipeline_compile_count &&
         left_allocations.pipeline_cache_hit_count ==
             right_allocations.pipeline_cache_hit_count &&
         left_allocations.pipeline_cache_eviction_count ==
             right_allocations.pipeline_cache_eviction_count &&
         left_allocations.descriptor_pool_create_count ==
             right_allocations.descriptor_pool_create_count &&
         left_allocations.descriptor_set_allocate_count ==
             right_allocations.descriptor_set_allocate_count &&
         left_allocations.descriptor_reuse_hit_count ==
             right_allocations.descriptor_reuse_hit_count &&
         left_allocations.buffer_allocation_count ==
             right_allocations.buffer_allocation_count &&
         left_allocations.buffer_reuse_hit_count ==
             right_allocations.buffer_reuse_hit_count &&
         lhs.outcome.failed_batches == rhs.outcome.failed_batches &&
         lhs.outcome.first_failed_batch == rhs.outcome.first_failed_batch &&
         lhs.outcome.first_status == rhs.outcome.first_status &&
         lhs.outcome.ok == rhs.outcome.ok &&
         std::string_view{lhs.outcome.reason} == rhs.outcome.reason;
}

} // namespace node_accel_contract::policy_case

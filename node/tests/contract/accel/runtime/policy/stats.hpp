#pragma once

#include <accel/runtime.hpp>

#include "local.hpp"

namespace node_accel_contract::policy_case {

[[nodiscard]] inline bool RuntimeStatsEqual(const rund::RuntimeStats &lhs,
                                            const rund::RuntimeStats &rhs) {
  return lhs.dispatch_count == rhs.dispatch_count &&
         lhs.command_submit_count == rhs.command_submit_count &&
         lhs.command_capacity == rhs.command_capacity &&
         lhs.command_inflight_peak == rhs.command_inflight_peak &&
         lhs.command_capacity_rejection_count ==
             rhs.command_capacity_rejection_count &&
         lhs.pipeline_compile_count == rhs.pipeline_compile_count &&
         lhs.pipeline_cache_hit_count == rhs.pipeline_cache_hit_count &&
         lhs.pipeline_cache_eviction_count ==
             rhs.pipeline_cache_eviction_count &&
         lhs.descriptor_pool_create_count == rhs.descriptor_pool_create_count &&
         lhs.descriptor_set_allocate_count ==
             rhs.descriptor_set_allocate_count &&
         lhs.descriptor_reuse_hit_count == rhs.descriptor_reuse_hit_count &&
         lhs.buffer_allocation_count == rhs.buffer_allocation_count &&
         lhs.buffer_reuse_hit_count == rhs.buffer_reuse_hit_count &&
         lhs.host_to_device_bytes == rhs.host_to_device_bytes &&
         lhs.device_to_host_bytes == rhs.device_to_host_bytes &&
         lhs.accel_kernel_ns == rhs.accel_kernel_ns &&
         lhs.accel_timestamp_count == rhs.accel_timestamp_count &&
         std::string_view{lhs.accel_timestamp_source} ==
             std::string_view{rhs.accel_timestamp_source} &&
         lhs.ok == rhs.ok && std::string_view{lhs.reason} == rhs.reason;
}

} // namespace node_accel_contract::policy_case

#pragma once

#include <accel/runtime.hpp>

namespace rund::node::accel::run_policy_detail {

[[nodiscard]] constexpr bool EvidenceOk(const RuntimeStats &stats) noexcept {
  return stats.outcome.ok &&
         (stats.run.work.dispatch_count != 0u ||
          stats.run.allocations.pipeline_compile_count != 0u ||
          stats.run.allocations.pipeline_cache_hit_count != 0u ||
          stats.run.allocations.pipeline_cache_eviction_count != 0u ||
          stats.run.allocations.descriptor_pool_create_count != 0u ||
          stats.run.allocations.descriptor_set_allocate_count != 0u ||
          stats.run.allocations.descriptor_reuse_hit_count != 0u ||
          stats.run.allocations.buffer_allocation_count != 0u ||
          stats.run.allocations.buffer_reuse_hit_count != 0u ||
          stats.run.transfer.host_to_device_bytes != 0u ||
          stats.run.transfer.device_to_host_bytes != 0u ||
          stats.run.time.accel_kernel_ns != 0u ||
          stats.run.time.accel_timestamp_count != 0u);
}

} // namespace rund::node::accel::run_policy_detail

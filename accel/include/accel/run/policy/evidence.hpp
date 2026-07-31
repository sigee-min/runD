#pragma once

#include <accel/runtime.hpp>

namespace rund::node::accel::run_policy_detail {

[[nodiscard]] constexpr bool EvidenceOk(const RuntimeStats &stats) noexcept {
  return stats.ok &&
         (stats.dispatch_count != 0u || stats.pipeline_compile_count != 0u ||
          stats.pipeline_cache_hit_count != 0u ||
          stats.pipeline_cache_eviction_count != 0u ||
          stats.descriptor_pool_create_count != 0u ||
          stats.descriptor_set_allocate_count != 0u ||
          stats.descriptor_reuse_hit_count != 0u ||
          stats.buffer_allocation_count != 0u ||
          stats.buffer_reuse_hit_count != 0u ||
          stats.host_to_device_bytes != 0u ||
          stats.device_to_host_bytes != 0u || stats.accel_kernel_ns != 0u ||
          stats.accel_timestamp_count != 0u);
}

} // namespace rund::node::accel::run_policy_detail

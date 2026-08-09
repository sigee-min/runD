#pragma once

#include <rund/compute/stats.hpp>

namespace rund::compute::detail {

// A sampled terminal is resident-clean only when execution reused the frozen
// Pipeline without construction, resource acquisition, host observation, or
// transfer activity. The sample epoch owns success and explicit-dirty state;
// this predicate owns the complete per-run Stats projection.
[[nodiscard]] constexpr bool
pipeline_sample_is_clean(const Stats &stats) noexcept {
  return stats.pipeline_compiles == 0u && stats.buffer_allocations == 0u &&
         stats.buffer_reuses == 0u && stats.download_events == 0u &&
         stats.uploaded_bytes == 0u && stats.downloaded_bytes == 0u &&
         stats.pipeline_cache_hits == 0u &&
         stats.pipeline_cache_evictions == 0u &&
         stats.descriptor_pool_creations == 0u &&
         stats.descriptor_set_allocations == 0u &&
         stats.descriptor_reuses == 0u &&
         stats.command_capacity_rejections == 0u &&
         stats.shader_compile_ns == 0u && stats.spirv_compile_ns == 0u &&
         stats.pipeline_create_ns == 0u && stats.descriptor_setup_ns == 0u &&
         stats.readback_ns == 0u && stats.output_hash == 0u &&
         stats.host_write_bytes == 0u &&
         stats.transfer_submissions.host_to_device == 0u &&
         stats.transfer_submissions.device_to_host == 0u &&
         stats.transfer_submissions.device_to_device == 0u;
}

} // namespace rund::compute::detail

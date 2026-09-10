#pragma once

#include "core.hpp"
#include "warm.hpp"

#include <cstdint>
#include <cstdio>

namespace rund::measure::compute {

void PrintWarm(const WarmCounters &warm);

template <class Stats> void PrintStats(const Stats &stats) {
  const auto field = [](const std::uint64_t value) {
    std::printf(",%llu", static_cast<unsigned long long>(value));
  };
  field(stats.pipeline_compiles);
  field(stats.buffer_allocations);
  field(stats.download_events);
  field(stats.dispatches);
  field(stats.command_submits);
  field(stats.uploaded_bytes);
  field(stats.downloaded_bytes);
  field(stats.pipeline_cache_hits);
  field(stats.pipeline_cache_evictions);
  field(stats.buffer_reuses);
  field(stats.descriptor_pool_creations);
  field(stats.descriptor_set_allocations);
  field(stats.descriptor_reuses);
  field(stats.original_dispatches);
  field(stats.final_dispatches);
  field(stats.fusions);
  field(stats.fusion_rejections);
  field(stats.internal_roundtrip_bytes);
  field(stats.external_roundtrip_bytes);
  field(stats.kernel_ns);
  field(stats.kernel_samples);
  field(stats.shader_compile_ns);
  field(stats.spirv_compile_ns);
  field(stats.pipeline_create_ns);
  field(stats.descriptor_setup_ns);
  field(stats.submit_wait_ns);
  field(stats.readback_ns);
  field(stats.graph_hash);
  field(stats.output_hash);
  field(stats.worker_count);
  field(stats.participating_workers);
  field(stats.tile_count);
  field(stats.tile_size);
  field(stats.vector_chunks);
  field(stats.tail_chunks);
}

} // namespace rund::measure::compute

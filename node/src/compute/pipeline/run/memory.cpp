#include "../local.hpp"
#include "../state.hpp"
#include "memory/internal.hpp"

#include <mutex>

namespace rund::compute::detail {

PipelineMemoryView pipeline_memory_view_locked(
    const PipelineState &state, const std::span<PipelineStepProfile> profiles,
    const bool include_residency_pool_buffers) noexcept {
  const PipelineSharedMemory shared =
      measure_pipeline_shared(state, include_residency_pool_buffers);
  const PipelineJobMemory jobs = measure_pipeline_jobs(
      state, shared, profiles, include_residency_pool_buffers);
  return PipelineMemoryView{.summary = jobs.summary,
                            .shared = shared.stats,
                            .scratch = shared.scratch,
                            .metadata = jobs.metadata,
                            .referenced_resource_bytes =
                                state.memory_inventory.referenced_resource_bytes,
                            .prepared = shared.prepared};
}

MemoryStats
pipeline_memory(const std::shared_ptr<PipelineState> &state) noexcept {
  if (!valid_pipeline(state)) {
    return {};
  }
  std::lock_guard lock{state->gate};
  return pipeline_memory_view_locked(*state).summary;
}

MemoryStats
pipeline_private_memory(const std::shared_ptr<PipelineState> &state) noexcept {
  if (!valid_pipeline(state)) {
    return {};
  }
  std::lock_guard lock{state->gate};
  return pipeline_memory_view_locked(*state, {}, false).summary;
}

} // namespace rund::compute::detail

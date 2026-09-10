#include "local.hpp"

#include "../../device/residency/pool.hpp"
#include "../../memory/local.hpp"
#include "../../pipeline/local.hpp"

#include <mutex>

namespace rund::compute::detail::virtual_observe {

MemoryStats virtual_memory_locked(const VirtualPipelineState &state) noexcept {
  MemoryStats memory{};
  if (state.geometry.route == VirtualRoute::GraphReduction ||
      state.geometry.route == VirtualRoute::GraphPointwise) {
    for (const std::shared_ptr<PipelineState> &pipeline :
         state.graph_pipelines) {
      merge_memory(memory, pipeline_private_memory(pipeline));
    }
    merge_memory(memory,
                 pipeline_private_memory(state.device_vsm_semantic_pipeline));
  } else {
    memory = pipeline_private_memory(state.pipeline);
    merge_memory(memory, pipeline_private_memory(state.alternate_pipeline));
  }
  const residency::Pool *const pool = state.pipeline->residency_pool.get();
  if (pool == nullptr) {
    return memory;
  }
  BufferMemory frames{};
  if (state.geometry.route == VirtualRoute::GraphReduction ||
      state.geometry.route == VirtualRoute::GraphPointwise) {
    for (const residency::PoolPhysicalOwner &owner : pool->graph_owners) {
      add_buffer_memory(frames, measure_buffers(owner.buffers));
    }
    add_buffer_memory(frames, measure_buffers(pool->control));
  } else {
    frames = measure_buffers(pool->input, pool->intermediate, pool->control,
                             pool->output);
  }
  merge_memory(memory.resident, fixed_memory(frames.resident));
  if (pool->device->backend == Backend::Cpu) {
    merge_memory(memory.host, fixed_memory(frames.physical, frames.reused));
  } else {
    merge_memory(memory.device, fixed_memory(frames.physical, frames.reused));
  }
  // These are the physical bytes of Authority-registered Host input/output
  // frames, not a transient backend transfer-scratch lease.
  merge_memory(memory.host, fixed_memory(pool->host_bytes));
  // Pool/arena/BufferState objects and prefetch arrays are logical retained
  // Host metadata. Their payload is already projected above and must not be
  // relabeled or physically re-accounted here.
  merge_memory(memory.host, fixed_memory(pool->retained_host_bytes()));
  return memory;
}

} // namespace rund::compute::detail::virtual_observe

namespace rund::compute::detail {

MemoryStats virtual_pipeline_memory(
    const std::shared_ptr<VirtualPipelineState> &state) noexcept {
  if (state == nullptr) {
    return {};
  }
  std::lock_guard lock{state->gate};
  return virtual_observe::virtual_memory_locked(*state);
}

} // namespace rund::compute::detail

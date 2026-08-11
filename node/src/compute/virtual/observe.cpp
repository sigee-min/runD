#include "state.hpp"

#include "../device/info.hpp"
#include "../device/residency_pool.hpp"
#include "../memory/local.hpp"
#include "../memory/profile.hpp"

#include <mutex>
#include <utility>

namespace rund::compute::detail {
namespace {

[[nodiscard]] MemoryStats
virtual_memory_locked(const VirtualPipelineState &state) noexcept {
  MemoryStats memory = pipeline_memory(state.pipeline);
  const residency::Pool *const pool = state.pipeline->residency_pool.get();
  if (pool == nullptr || pool->cache_input == nullptr ||
      pool->cache_output == nullptr) {
    return {};
  }
  BufferMemory cache = measure_buffer(pool->cache_input);
  add_buffer_memory(cache, measure_buffer(pool->cache_output));
  merge_memory(memory.resident, fixed_memory(cache.resident));
  if (state.pipeline->device->backend == Backend::Cpu) {
    merge_memory(memory.host, fixed_memory(cache.physical, cache.reused));
  } else {
    merge_memory(memory.device, fixed_memory(cache.physical, cache.reused));
  }
  merge_memory(memory.staging, fixed_memory(pool->host_bytes));
  return memory;
}

} // namespace

bool valid_virtual_pipeline(
    const std::shared_ptr<VirtualPipelineState> &state) noexcept {
  return state != nullptr && state->input != nullptr &&
         state->output != nullptr && state->pipeline != nullptr &&
         state->pipeline->residency != nullptr &&
         state->pipeline->residency_pool != nullptr &&
         state->pipeline->residency->identity() &&
         valid_pipeline(state->pipeline);
}

Stats virtual_pipeline_stats(
    const std::shared_ptr<VirtualPipelineState> &state) noexcept {
  if (state == nullptr) {
    return {};
  }
  std::lock_guard lock{state->gate};
  return state->stats;
}

MemoryStats virtual_pipeline_memory(
    const std::shared_ptr<VirtualPipelineState> &state) noexcept {
  if (state == nullptr) {
    return {};
  }
  std::lock_guard lock{state->gate};
  return virtual_memory_locked(*state);
}

PipelinePlan virtual_pipeline_plan(
    const std::shared_ptr<VirtualPipelineState> &state) noexcept {
  if (state == nullptr) {
    return {};
  }
  std::lock_guard lock{state->gate};
  return pipeline_plan(state->pipeline);
}

Result<telemetry::Profile> virtual_pipeline_profile(
    const std::shared_ptr<VirtualPipelineState> &state) noexcept {
  if (!valid_virtual_pipeline(state) || state->pipeline->device == nullptr) {
    return Result<telemetry::Profile>::fail(Reason::ProfileInvalid);
  }
  std::unique_lock lock{state->gate, std::try_to_lock};
  if (!lock.owns_lock() || state->phase == VirtualPipelinePhase::Running) {
    return Result<telemetry::Profile>::fail(Reason::ProfileBusy);
  }
  if (state->samples != VirtualPipelineState::SampleState::Inactive) {
    return Result<telemetry::Profile>::fail(Reason::ProfileBusy);
  }
  std::shared_ptr<const DeviceInfo> device =
      device_info_owner(state->pipeline->device);
  if (device == nullptr) {
    return Result<telemetry::Profile>::fail(Reason::DeviceInfoInvalid);
  }
  return Result<telemetry::Profile>::success(ProfileAccess::make(
      std::move(device), state->stats, virtual_memory_locked(*state)));
}

} // namespace rund::compute::detail

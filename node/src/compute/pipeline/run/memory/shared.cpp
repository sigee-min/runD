#include "internal.hpp"

#include "../../state.hpp"

#include "../../../backend.hpp"
#include "../../../device/residency/pool.hpp"
#include "../../../memory/cpu.hpp"
#include "../../../memory/local.hpp"

#include "../../../../accel/kernel/prepared/interface/api.hpp"

#include <rund/counter.hpp>

#include <algorithm>
#include <array>
#include <cstdint>
#include <limits>
#include <span>

namespace rund::compute::detail {
namespace {

[[nodiscard]] std::uint64_t
base_host_bytes(const PipelineState &state) noexcept {
  std::uint64_t bytes = ::rund::detail::counter::SaturatingAdd(
      sizeof(PipelineState), vector_memory(state.memory_inventory.jobs));
  if (state.publication != nullptr) {
    bytes = ::rund::detail::counter::SaturatingAdd(
        bytes, sizeof(PipelinePublicationState));
    bytes = ::rund::detail::counter::SaturatingAdd(
        bytes, vector_memory(state.publication->state_pairs));
  }
  bytes =
      ::rund::detail::counter::SaturatingAdd(bytes, vector_memory(state.steps));
  bytes = ::rund::detail::counter::SaturatingAdd(
      bytes, vector_memory(state.window_rank));
  bytes = ::rund::detail::counter::SaturatingAdd(bytes,
                                                 state.windows.retained_bytes());
  bytes = ::rund::detail::counter::SaturatingAdd(
      bytes, vector_memory(state.resources));
  bytes = ::rund::detail::counter::SaturatingAdd(
      bytes, vector_memory(state.prepared_buffers));
  bytes = ::rund::detail::counter::SaturatingAdd(
      bytes, vector_memory(state.cpu_storage));
  bytes = ::rund::detail::counter::SaturatingAdd(bytes,
                                                 vector_memory(state.claims));
  bytes = ::rund::detail::counter::SaturatingAdd(
      bytes, vector_memory(state.alternate_claims));
  bytes = ::rund::detail::counter::SaturatingAdd(
      bytes, vector_memory(state.publications));
  bytes = ::rund::detail::counter::SaturatingAdd(bytes,
                                                 vector_memory(state.outputs));
  bytes = ::rund::detail::counter::SaturatingAdd(
      bytes, vector_memory(state.output_lookup));
  bytes = ::rund::detail::counter::SaturatingAdd(
      bytes, vector_memory(state.dependencies));
  bytes = ::rund::detail::counter::SaturatingAdd(bytes,
                                                 vector_memory(state.barriers));
  if (state.profile != nullptr) {
    bytes = ::rund::detail::counter::SaturatingAdd(
        bytes, sizeof(PipelineProfileState));
    bytes = ::rund::detail::counter::SaturatingAdd(
        bytes, vector_memory(state.profile->steps));
    bytes = ::rund::detail::counter::SaturatingAdd(
        bytes, vector_memory(state.profile->started_ns));
    bytes = ::rund::detail::counter::SaturatingAdd(
        bytes, vector_memory(state.profile->started));
  }
  return bytes;
}

} // namespace

static PipelineSharedMemory project_pipeline_shared_owners(
    const PipelineState &state,
    const bool include_residency_pool_buffers) noexcept {
  MemoryStats shared{
      .backend = state.device->backend,
      .scope = MemoryScope::Pipeline,
  };
  std::uint64_t metadata = base_host_bytes(state);
  shared.host = fixed_memory(metadata);
  std::array<const JobWorkspace *, PipelineRouteCapacity> shared_workspaces;
  std::size_t shared_workspace_count = 0u;
  const CpuPreparedArena *const shared_cpu_arena =
      state.cpu_prepared_arena.get();
  bool invalid_shared_owners = false;
  const auto include_buffer =
      [&](const std::shared_ptr<BufferState> &buffer) -> bool {
    return include_residency_pool_buffers ||
           !pipeline_pool_owns_buffer(state.residency_pool.get(), buffer);
  };
  const auto collect_cpu_arena = [&](const CpuPreparedArena *const arena) {
    if (arena == nullptr || shared_cpu_arena == nullptr ||
        shared_cpu_arena != arena) {
      invalid_shared_owners = true;
    }
  };
  std::array<const JobArena *, PipelineRouteCapacity> measured_arenas;
  std::size_t measured_arena_count = 0u;
  std::uint64_t arena_metadata = 0u;
  const auto measure_arena = [&](const JobArena *const arena) {
    if (arena == nullptr) {
      return;
    }
    const auto end = measured_arenas.begin() + measured_arena_count;
    if (std::find(measured_arenas.begin(), end, arena) != end) {
      return;
    }
    if (measured_arena_count == measured_arenas.size()) {
      invalid_shared_owners = true;
      return;
    }
    measured_arenas[measured_arena_count++] = arena;
    ::rund::detail::counter::Accumulate(arena_metadata, sizeof(JobArena));
    ::rund::detail::counter::Accumulate(arena_metadata,
                                        vector_memory(arena->buffers));
    ::rund::detail::counter::Accumulate(arena_metadata,
                                        vector_memory(arena->slots));
    ::rund::detail::counter::Accumulate(arena_metadata,
                                        vector_memory(arena->scratch));
    ::rund::detail::counter::Accumulate(
        arena_metadata, vector_memory(arena->binds.overflow_refs));
    ::rund::detail::counter::Accumulate(
        arena_metadata, vector_memory(arena->binds.overflow_handles));
  };
  const auto collect_shared = [&](const std::shared_ptr<JobState> &job) {
    if (job == nullptr) {
      return;
    }
    if (state.device->backend == Backend::Cpu) {
      collect_cpu_arena(job->cpu_prepared_arena.get());
    } else if (job->cpu_prepared_arena != nullptr) {
      invalid_shared_owners = true;
    }
    if (job->cpu != nullptr && job->cpu->graph != nullptr) {
      const std::shared_ptr<CpuGraphStorage> &storage =
          job->cpu->graph->storage;
      const bool storage_owned =
          storage != nullptr &&
          std::find(state.cpu_storage.begin(), state.cpu_storage.end(),
                    storage) != state.cpu_storage.end();
      if (!storage_owned || job->cpu_prepared_arena == nullptr ||
          storage->prepared_arena != job->cpu_prepared_arena) {
        invalid_shared_owners = true;
      }
    }
    if (job->workspace == nullptr) {
      return;
    }
    const JobWorkspace *const workspace = job->workspace.get();
    const auto end = shared_workspaces.begin() + shared_workspace_count;
    if (std::find(shared_workspaces.begin(), end, workspace) != end) {
      return;
    }
    if (shared_workspace_count == shared_workspaces.size()) {
      invalid_shared_owners = true;
      return;
    }
    shared_workspaces[shared_workspace_count++] = workspace;
    measure_arena(workspace->arena.get());
  };
  for (const PipelineStep &step : state.steps) {
    collect_shared(step.job);
    collect_shared(step.alternate_job);
  }
  for (std::size_t index = 0u; index < state.cpu_storage.size(); ++index) {
    const std::shared_ptr<CpuGraphStorage> &storage = state.cpu_storage[index];
    if (storage == nullptr ||
        std::find(state.cpu_storage.begin(), state.cpu_storage.begin() + index,
                  storage) != state.cpu_storage.begin() + index) {
      invalid_shared_owners = true;
      continue;
    }
    collect_cpu_arena(storage->prepared_arena.get());
  }
  if ((state.device->backend == Backend::Cpu) !=
          (shared_cpu_arena != nullptr) ||
      (state.device->backend != Backend::Cpu && !state.cpu_storage.empty())) {
    invalid_shared_owners = true;
  }
  if (shared_cpu_arena != nullptr) {
    CpuStorageBytes cpu = cpu_prepared_arena_memory(shared_cpu_arena);
    cpu.host = add_cpu_memory_bytes(cpu.host, sizeof(CpuPreparedArena));
    ::rund::detail::counter::Accumulate(metadata, cpu.host);
    merge_memory(shared.host, fixed_memory(cpu.host));
    merge_memory(shared.tile, fixed_memory(cpu.tile));
  }
  // PipelineState::cpu_storage is the canonical Program-storage owner list.
  // Charge each entry once at that shared boundary; Job lifetime references
  // cannot change attribution by traversal order.
  for (const std::shared_ptr<CpuGraphStorage> &storage : state.cpu_storage) {
    if (storage == nullptr) {
      continue;
    }
    const CpuStorageBytes cpu = cpu_graph_storage_private_memory(storage.get());
    ::rund::detail::counter::Accumulate(metadata, cpu.host);
    merge_memory(shared.host, fixed_memory(cpu.host));
    merge_memory(shared.tile, fixed_memory(cpu.tile));
  }
  const std::span<const std::shared_ptr<BufferState>> prepared_buffers{
      state.prepared_buffers};
  const std::size_t scratch_count =
      state.plan.scratch_count <= prepared_buffers.size()
          ? static_cast<std::size_t>(state.plan.scratch_count)
          : 0u;
  const BufferMemory scratch =
      measure_buffers(prepared_buffers.last(scratch_count));
  BufferMemory shared_buffers{};
  for (std::size_t workspace_index = 0u;
       workspace_index < shared_workspace_count; ++workspace_index) {
    const auto &buffers = shared_workspaces[workspace_index]->buffers;
    for (std::size_t buffer_index = 0u; buffer_index < buffers.size();
         ++buffer_index) {
      const std::shared_ptr<BufferState> &buffer = buffers[buffer_index];
      bool first_owner = true;
      for (std::size_t prior_workspace = 0u;
           prior_workspace <= workspace_index && first_owner;
           ++prior_workspace) {
        const auto &prior = shared_workspaces[prior_workspace]->buffers;
        const std::size_t prior_count =
            prior_workspace == workspace_index ? buffer_index : prior.size();
        const auto prior_end =
            prior_count == 0u ? prior.begin() : prior.begin() + prior_count;
        first_owner = std::find(prior.begin(), prior_end, buffer) == prior_end;
      }
      if (first_owner && include_buffer(buffer)) {
        add_buffer_memory(shared_buffers, measure_buffer(buffer));
      }
    }
  }
  add_buffer_memory(shared_buffers,
                    measure_buffers(prepared_buffers.first(
                        prepared_buffers.size() - scratch_count)));
  if (invalid_shared_owners) {
    shared_buffers = BufferMemory{
        .resident = std::numeric_limits<std::uint64_t>::max(),
        .physical = std::numeric_limits<std::uint64_t>::max(),
        .reused = std::numeric_limits<std::uint64_t>::max(),
    };
  }
  add_buffer_memory(shared_buffers, scratch);
  BufferMemory owned_buffers{};
  for (const PipelineResource &resource : state.resources) {
    if (resource.owned && include_buffer(resource.buffer)) {
      add_buffer_memory(owned_buffers, measure_buffer(resource.buffer));
    }
  }
  add_buffer_memory(owned_buffers, shared_buffers);
  merge_memory(shared.resident, fixed_memory(owned_buffers.resident));
  if (state.device->backend == Backend::Cpu) {
    merge_memory(shared.host,
                 fixed_memory(owned_buffers.physical, owned_buffers.reused));
  } else {
    merge_memory(shared.device,
                 fixed_memory(owned_buffers.physical, owned_buffers.reused));
  }

  ::rund::detail::counter::Accumulate(metadata, arena_metadata);
  merge_memory(shared.host, fixed_memory(arena_metadata));

  if (invalid_shared_owners) {
    constexpr std::uint64_t invalid = std::numeric_limits<std::uint64_t>::max();
    metadata = invalid;
    shared.host = fixed_memory(invalid, invalid);
    shared.tile = fixed_memory(invalid, invalid);
    shared.resident = fixed_memory(invalid, invalid);
    shared.device = fixed_memory(invalid, invalid);
  }
  return PipelineSharedMemory{.stats = shared,
                              .scratch = scratch,
                              .metadata = metadata,
                              .prepared = {}};
}

void seal_pipeline_shared(PipelineState &state) noexcept {
  auto &inventory = state.memory_inventory;
  const auto inclusive = project_pipeline_shared_owners(state, true);
  const auto ownership = [](const MemoryStats &stats) noexcept {
    return PipelineSharedOwnership{.host = stats.host,
                                   .device = stats.device,
                                   .tile = stats.tile,
                                   .resident = stats.resident};
  };
  inventory.shared[0u] = ownership(inclusive.stats);
  inventory.shared[1u] =
      state.residency_pool == nullptr
          ? inventory.shared[0u]
          : ownership(project_pipeline_shared_owners(state, false).stats);
  inventory.metadata = inclusive.metadata;
  inventory.cpu_storage_count = state.cpu_storage.size();
  inventory.scratch_resident = inclusive.scratch.resident;
  inventory.scratch_physical = inclusive.scratch.physical;
  inventory.scratch_reused = inclusive.scratch.reused;
  inventory.referenced_resource_bytes = 0u;
  for (const PipelineResource &resource : state.resources) {
    if (!resource.owned) {
      ::rund::detail::counter::Accumulate(inventory.referenced_resource_bytes,
                                          resource.bytes);
    }
  }
}

PipelineSharedMemory
measure_pipeline_shared(const PipelineState &state,
                        const bool include_residency_pool_buffers) noexcept {
  const auto &inventory = state.memory_inventory;
  const auto &owned =
      inventory.shared[include_residency_pool_buffers ? 0u : 1u];
  MemoryStats shared{.backend = state.device->backend,
                     .scope = MemoryScope::Pipeline,
                     .host = owned.host,
                     .tile = owned.tile,
                     .resident = owned.resident,
                     .device = owned.device};
  shared.frame = MemoryCounter{.current = state.frame_current,
                               .peak = state.frame_peak,
                               .cumulative = state.frame_bytes,
                               .reused = state.frame_reused,
                               .budget = state.frame_budget};
  shared.transfer = MemoryCounter{.peak = state.transfer_peak,
                                  .cumulative = state.transfer_bytes};
  node::accel::detail::PreparedPipelineMemory prepared{};
  if (state.device->backend != Backend::Cpu) {
    prepared = state.device->ops != nullptr &&
                       state.device->ops->pipeline_memory != nullptr
                   ? state.device->ops->pipeline_memory(state)
                   : node::accel::detail::PreparedPipelineMemory{};
    merge_memory(shared.host, pipeline_prepared_memory(prepared.host));
    merge_memory(shared.device, pipeline_prepared_memory(prepared.device));
    merge_memory(shared.staging, pipeline_prepared_memory(prepared.staging));
  }
  shared.staging.cumulative = ::rund::detail::counter::SaturatingAdd(
      shared.staging.cumulative, state.staging_bytes);
  shared.staging.reused = ::rund::detail::counter::SaturatingAdd(
      shared.staging.reused, state.staging_reused);
  shared.staging.budget = std::max(shared.staging.budget, state.staging_budget);
  shared.staging.peak = std::max(
      shared.staging.peak, ::rund::detail::counter::SaturatingAdd(
                               shared.staging.current, state.staging_peak));

  // Validate sealed ownership coordinates without reconstructing ownership.
  const auto *arena = state.cpu_prepared_arena.get();
  bool invalid =
      (state.device->backend == Backend::Cpu) != (arena != nullptr) ||
      state.cpu_storage.size() != inventory.cpu_storage_count;
  for (const auto &storage : state.cpu_storage) {
    invalid =
        invalid || storage == nullptr || storage->prepared_arena.get() != arena;
  }
  for (const auto &owner : inventory.jobs) {
    const auto &step = state.steps[owner.ordinal / 2u];
    const auto &job = owner.ordinal % 2u == 0u ? step.job : step.alternate_job;
    if (job == nullptr) {
      invalid = true;
      continue;
    }
    invalid = invalid || job->cpu_prepared_arena.get() != arena;
    if (job->cpu != nullptr && job->cpu->graph != nullptr) {
      invalid =
          invalid || owner.cpu_storage >= state.cpu_storage.size() ||
          job->cpu->graph->storage != state.cpu_storage[owner.cpu_storage];
    }
  }
  if (invalid) {
    constexpr auto maximum = std::numeric_limits<std::uint64_t>::max();
    shared.host = shared.tile = shared.resident = shared.device =
        fixed_memory(maximum, maximum);
  }
  return {.stats = shared,
          .scratch = {.resident = inventory.scratch_resident,
                      .physical = inventory.scratch_physical,
                      .reused = inventory.scratch_reused},
          .metadata = invalid ? std::numeric_limits<std::uint64_t>::max()
                              : inventory.metadata,
          .prepared = prepared};
}

} // namespace rund::compute::detail

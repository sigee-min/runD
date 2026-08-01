#include <rund/compute/pipeline.hpp>

#include "../../../accel/kernel/prepared.hpp"
#include "../../backend.hpp"
#include "../../memory/cpu.hpp"
#include "../../memory/local.hpp"
#include "../claim.hpp"
#include "../local.hpp"
#include "../state.hpp"
#include "clock.hpp"

#include <rund/counter.hpp>

#include <algorithm>
#include <array>
#include <cstdint>
#include <mutex>
#include <span>

namespace rund::compute::detail {
namespace {

[[nodiscard]] std::uint64_t
base_host_bytes(const PipelineState &state) noexcept {
  std::uint64_t bytes = sizeof(PipelineState);
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
                                                 vector_memory(state.windows));
  bytes = ::rund::detail::counter::SaturatingAdd(
      bytes, vector_memory(state.resources));
  bytes = ::rund::detail::counter::SaturatingAdd(
      bytes, vector_memory(state.shared_buffers));
  bytes = ::rund::detail::counter::SaturatingAdd(
      bytes, vector_memory(state.prepared_buffers));
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
  }
  return bytes;
}

[[nodiscard]] std::span<const std::shared_ptr<BufferState>>
internal_buffers(const JobState &job) noexcept {
  if (job.workspace != nullptr) {
    return job.workspace->buffers;
  }
  if (job.cpu != nullptr && job.cpu->graph != nullptr) {
    return job.cpu->graph->owned_buffers;
  }
  return job.graph_buffers;
}

[[nodiscard]] MemoryCounter
prepared_memory(const node::accel::detail::PreparedMemory memory) noexcept {
  return MemoryCounter{.current = memory.current,
                       .peak = memory.peak,
                       .cumulative = memory.cumulative,
                       .reused = memory.reused,
                       .budget = memory.budget};
}

struct MemoryView final {
  MemoryStats summary{};
  MemoryStats shared{};
  BufferMemory scratch{};
  std::uint64_t metadata{};
  std::uint64_t referenced_resource_bytes{};
  node::accel::detail::PreparedPipelineMemory prepared{};
};

[[nodiscard]] MemoryCounter without(const MemoryCounter total,
                                    const MemoryCounter part) noexcept {
  return MemoryCounter{
      .current =
          total.current >= part.current ? total.current - part.current : 0u,
      .peak = total.peak >= part.peak ? total.peak - part.peak : 0u,
      .cumulative = total.cumulative >= part.cumulative
                        ? total.cumulative - part.cumulative
                        : 0u,
      .reused = total.reused >= part.reused ? total.reused - part.reused : 0u,
      .budget = total.budget >= part.budget ? total.budget - part.budget : 0u,
  };
}

[[nodiscard]] MemoryView
measure(const PipelineState &state,
        const std::span<PipelineStepProfile> profiles = {}) noexcept {
  MemoryStats shared{
      .backend = state.device->backend,
      .scope = MemoryScope::Pipeline,
      .frame = MemoryCounter{.current = state.frame_current,
                             .peak = state.frame_peak,
                             .cumulative = state.frame_bytes,
                             .reused = state.frame_reused,
                             .budget = state.frame_budget},
      .transfer = MemoryCounter{.peak = state.read_transfer_peak,
                                .cumulative = state.read_transfer_bytes,
                                .budget = device_budget(state.device)},
  };
  std::uint64_t metadata = base_host_bytes(state);
  shared.host = fixed_memory(metadata);
  node::accel::detail::PreparedPipelineMemory prepared{};
  if (state.device->backend != Backend::Cpu) {
    prepared = state.device->ops != nullptr &&
                       state.device->ops->pipeline_memory != nullptr
                   ? state.device->ops->pipeline_memory(state)
                   : node::accel::detail::PreparedPipelineMemory{};
    merge_memory(shared.host, prepared_memory(prepared.host));
    merge_memory(shared.device, prepared_memory(prepared.device));
    merge_memory(shared.staging, prepared_memory(prepared.staging));
  }
  const std::span<const std::shared_ptr<BufferState>> prepared_buffers{
      state.prepared_buffers};
  const std::size_t scratch_count =
      state.plan.scratch_count <= prepared_buffers.size()
          ? static_cast<std::size_t>(state.plan.scratch_count)
          : 0u;
  const BufferMemory scratch =
      measure_buffers(prepared_buffers.last(scratch_count));
  BufferMemory shared_buffers = measure_buffers(
      state.shared_buffers,
      prepared_buffers.first(prepared_buffers.size() - scratch_count));
  add_buffer_memory(shared_buffers, scratch);
  BufferMemory owned_buffers{};
  for (const PipelineResource &resource : state.resources) {
    if (resource.owned) {
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
  shared.staging.cumulative = ::rund::detail::counter::SaturatingAdd(
      shared.staging.cumulative, state.read_staging_bytes);
  shared.staging.reused = ::rund::detail::counter::SaturatingAdd(
      shared.staging.reused, state.read_staging_reused);
  shared.staging.budget =
      std::max(shared.staging.budget, state.read_staging_budget);
  shared.staging.peak =
      std::max(shared.staging.peak,
               ::rund::detail::counter::SaturatingAdd(shared.staging.current,
                                                      state.read_staging_peak));

  std::array<const JobArena *, PipelineStepCapacity> measured_arenas{};
  std::size_t measured_arena_count = 0u;
  std::uint64_t arena_metadata = 0u;
  const auto measure_arena = [&](const std::shared_ptr<JobState> &job) {
    if (job == nullptr || job->workspace == nullptr ||
        job->workspace->arena == nullptr) {
      return;
    }
    const JobArena *const arena = job->workspace->arena.get();
    const auto end = measured_arenas.begin() + measured_arena_count;
    if (std::find(measured_arenas.begin(), end, arena) != end) {
      return;
    }
    measured_arenas[measured_arena_count++] = arena;
    ::rund::detail::counter::Accumulate(arena_metadata, sizeof(JobArena));
    ::rund::detail::counter::Accumulate(arena_metadata,
                                        vector_memory(arena->buffers));
    ::rund::detail::counter::Accumulate(arena_metadata,
                                        vector_memory(arena->slots));
    ::rund::detail::counter::Accumulate(
        arena_metadata, vector_memory(arena->binds.overflow_refs));
    ::rund::detail::counter::Accumulate(
        arena_metadata, vector_memory(arena->binds.overflow_handles));
  };
  for (const PipelineStep &step : state.steps) {
    measure_arena(step.job);
    measure_arena(step.alternate_job);
  }
  ::rund::detail::counter::Accumulate(metadata, arena_metadata);
  merge_memory(shared.host, fixed_memory(arena_metadata));

  MemoryStats memory = shared;
  std::array<const JobState *, PipelineIterationCapacity * 2u> measured_jobs{};
  std::size_t measured_job_count = 0u;
  std::array<const JobWorkspace *, PipelineStepCapacity> measured_workspaces{};
  std::size_t measured_workspace_count = 0u;
  for (std::size_t index = 0u; index < state.steps.size(); ++index) {
    const PipelineStep &step = state.steps[index];
    MemoryStats owned{.backend = state.device->backend,
                      .scope = MemoryScope::Pipeline};
    std::uint64_t owned_metadata{};
    MemoryCounter owned_internal_host{};
    const auto measure_job = [&](const std::shared_ptr<JobState> &job) {
      if (job == nullptr) {
        return;
      }
      const JobState *const identity = job.get();
      if (std::find(measured_jobs.begin(),
                    measured_jobs.begin() + measured_job_count,
                    identity) != measured_jobs.begin() + measured_job_count) {
        return;
      }
      measured_jobs[measured_job_count++] = identity;
      ::rund::detail::counter::Accumulate(owned_metadata, sizeof(JobState));
      ::rund::detail::counter::Accumulate(owned_metadata,
                                          vector_memory(job->inputs));
      ::rund::detail::counter::Accumulate(owned_metadata,
                                          vector_memory(job->input_views));
      ::rund::detail::counter::Accumulate(owned_metadata,
                                          vector_memory(job->write_inputs));
      ::rund::detail::counter::Accumulate(owned_metadata,
                                          vector_memory(job->graph_buffers));
      ::rund::detail::counter::Accumulate(owned_metadata,
                                          vector_memory(job->outputs));
      ::rund::detail::counter::Accumulate(owned_metadata,
                                          vector_memory(job->output_views));
      ::rund::detail::counter::Accumulate(owned_metadata,
                                          vector_memory(job->views));
      ::rund::detail::counter::Accumulate(owned_metadata,
                                          vector_memory(job->cpu_view_inputs));
      ::rund::detail::counter::Accumulate(owned_metadata,
                                          vector_memory(job->cpu_view_outputs));
      ::rund::detail::counter::Accumulate(owned_metadata,
                                          vector_memory(job->cpu_view_buffers));
      const CpuRetainedMemory cpu = cpu_run_memory(job->cpu.get());
      ::rund::detail::counter::Accumulate(owned_metadata, cpu.host);
      bool measure_internal = true;
      if (job->workspace != nullptr) {
        const JobWorkspace *const workspace = job->workspace.get();
        const auto end = measured_workspaces.begin() + measured_workspace_count;
        if (std::find(measured_workspaces.begin(), end, workspace) != end) {
          measure_internal = false;
        } else {
          measured_workspaces[measured_workspace_count++] = workspace;
          ::rund::detail::counter::Accumulate(owned_metadata,
                                              sizeof(JobWorkspace));
          ::rund::detail::counter::Accumulate(
              owned_metadata, vector_memory(workspace->buffers));
          ::rund::detail::counter::Accumulate(
              owned_metadata, vector_memory(workspace->offsets));
        }
      }
      BufferMemory internal{};
      if (measure_internal) {
        for (const std::shared_ptr<BufferState> &buffer :
             internal_buffers(*job)) {
          if (std::find(state.shared_buffers.begin(),
                        state.shared_buffers.end(),
                        buffer) == state.shared_buffers.end()) {
            add_buffer_memory(internal, measure_buffer(buffer));
          }
        }
      }
      const BufferMemory view_buffers = measure_buffers(job->cpu_view_buffers);
      merge_memory(owned.resident, fixed_memory(internal.resident));
      merge_memory(owned.resident, fixed_memory(view_buffers.resident));
      if (state.device->backend == Backend::Cpu) {
        merge_memory(owned_internal_host,
                     fixed_memory(internal.physical, internal.reused));
        merge_memory(owned_internal_host,
                     fixed_memory(view_buffers.physical, view_buffers.reused));
        merge_memory(owned.tile, fixed_memory(cpu.tile));
      } else {
        merge_memory(owned.device,
                     fixed_memory(internal.physical, internal.reused));
        if (state.device->ops != nullptr &&
            state.device->ops->job_staging != nullptr) {
          merge_memory(owned.staging, state.device->ops->job_staging(*job));
        }
      }
    };
    measure_job(step.job);
    measure_job(step.alternate_job);
    owned.host = fixed_memory(owned_metadata);
    merge_memory(owned.host, owned_internal_host);
    ::rund::detail::counter::Accumulate(metadata, owned_metadata);
    merge_memory(memory, owned);
    if (index < profiles.size()) {
      profiles[index].memory = owned;
    }
  }
  std::uint64_t referenced_resource_bytes{};
  for (const PipelineResource &resource : state.resources) {
    if (!resource.owned) {
      ::rund::detail::counter::Accumulate(referenced_resource_bytes,
                                          resource.bytes);
    }
  }
  return MemoryView{.summary = memory,
                    .shared = shared,
                    .scratch = scratch,
                    .metadata = metadata,
                    .referenced_resource_bytes = referenced_resource_bytes,
                    .prepared = prepared};
}

} // namespace

MemoryStats
pipeline_memory(const std::shared_ptr<PipelineState> &state) noexcept {
  if (!valid_pipeline(state)) {
    return {};
  }
  std::lock_guard lock{state->gate};
  return measure(*state).summary;
}

MemorySnapshot
pipeline_memory_snapshot(const std::shared_ptr<PipelineState> &state,
                         const std::span<MemoryEntry> entries) noexcept {
  if (!valid_pipeline(state)) {
    SnapshotWriter writer{MemoryStats{}, entries};
    return writer.finish();
  }
  std::lock_guard lock{state->gate};
  const MemoryView view = measure(*state);
  const MemoryStats &summary = view.summary;
  SnapshotWriter writer{summary, entries};
  const std::uint64_t metadata = view.metadata;
  writer.add(MemoryCategory::Host, MemoryUse::Metadata, 0u,
             fixed_memory(metadata));
  if (state->device->backend == Backend::Cpu &&
      summary.host.current > metadata) {
    writer.add(
        MemoryCategory::Host, MemoryUse::Internal, 0u,
        fixed_memory(summary.host.current - metadata, summary.host.reused));
  } else if (state->device->backend != Backend::Cpu) {
    const MemoryCounter native_host = prepared_memory(view.prepared.host);
    if (native_host.current != 0u || native_host.cumulative != 0u) {
      writer.add(MemoryCategory::Host, MemoryUse::Metadata, 1u, native_host);
    }
  }
  if (summary.resident.current != 0u) {
    const MemoryCounter scratch = fixed_memory(view.scratch.resident);
    const MemoryCounter internal = without(summary.resident, scratch);
    if (internal.current != 0u || internal.cumulative != 0u) {
      writer.add(MemoryCategory::Resident, MemoryUse::Internal, 0u, internal);
    }
    if (scratch.current != 0u) {
      writer.add(MemoryCategory::Resident, MemoryUse::Scratch, 0u, scratch);
    }
  }
  if (summary.device.current != 0u) {
    const MemoryCounter scratch =
        fixed_memory(view.scratch.physical, view.scratch.reused);
    const MemoryCounter internal = without(summary.device, scratch);
    if (internal.current != 0u || internal.cumulative != 0u) {
      writer.add(MemoryCategory::Device, MemoryUse::Internal, 0u, internal);
    }
    if (scratch.current != 0u) {
      writer.add(MemoryCategory::Device, MemoryUse::Scratch, 0u, scratch);
    }
  }
  if (summary.tile.current != 0u) {
    writer.add(MemoryCategory::Tile, MemoryUse::Scratch, 0u, summary.tile);
  }
  if (summary.staging.current != 0u) {
    writer.add(MemoryCategory::Staging, MemoryUse::Coordinator, 0u,
               summary.staging);
  }
  if (summary.frame.current != 0u || summary.frame.cumulative != 0u) {
    writer.add(MemoryCategory::Frame, MemoryUse::Coordinator, 0u,
               summary.frame);
  }
  if (summary.transfer.peak != 0u || summary.transfer.cumulative != 0u) {
    writer.add(MemoryCategory::Transfer, MemoryUse::Traffic, 0u,
               summary.transfer);
  }
  return writer.finish();
}

Result<PipelineProfileSnapshot>
pipeline_profile(const std::shared_ptr<PipelineState> &state,
                 const std::span<PipelineStepProfile> steps) noexcept {
  if (!valid_pipeline(state)) {
    return Result<PipelineProfileSnapshot>::fail(Reason::ProfileInvalid);
  }
  std::unique_lock lock{state->gate, std::try_to_lock};
  if (!lock.owns_lock() || state->phase == PipelinePhase::Running) {
    return Result<PipelineProfileSnapshot>::fail(Reason::ProfileBusy);
  }
  if (state->profile == nullptr) {
    return Result<PipelineProfileSnapshot>::fail(Reason::ProfileUnavailable);
  }
  if (state->publication != nullptr) {
    std::lock_guard publication_lock{state->publication->gate};
    synchronize_pipeline_observation_epoch(*state, *state->publication);
    state->stats.publication.generation = state->publication->generation;
  }
  const std::uint64_t started = pipeline_clock();
  const std::span<PipelineStepProfile> canonical{state->profile->steps.data(),
                                                 state->steps.size()};
  const MemoryView view = measure(*state, canonical);
  const std::size_t written = std::min(steps.size(), canonical.size());
  std::copy_n(canonical.begin(), written, steps.begin());
  PipelineProfileSnapshot snapshot{
      .execution = state->stats,
      .memory = view.summary,
      .shared_memory = view.shared,
      .referenced_resource_bytes = view.referenced_resource_bytes,
      .instrumentation_command_count =
          state->profile->instrumentation_command_count,
      .instrumentation_byte_count = state->profile->instrumentation_byte_count,
      .written = written,
      .total = canonical.size(),
  };
  const std::uint64_t finished = pipeline_clock();
  snapshot.observation =
      StepTiming{.duration_ns = finished >= started ? finished - started : 0u,
                 .sample_count = 1u,
                 .clock = StepClock::HostSteady,
                 .relation = StepTimingRelation::Exclusive};
  return Result<PipelineProfileSnapshot>::success(snapshot);
}

} // namespace rund::compute::detail

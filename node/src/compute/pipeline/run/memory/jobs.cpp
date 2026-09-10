#include "internal.hpp"

#include "../../state.hpp"

#include "../../../backend.hpp"
#include "../../../memory/local.hpp"

#include <rund/counter.hpp>

#include <algorithm>
#include <array>
#include <bitset>
#include <cstdint>
#include <functional>
#include <limits>

namespace rund::compute::detail {
namespace {

[[nodiscard]] std::span<const std::shared_ptr<BufferState>>
internal_buffers(const JobState &job) noexcept {
  if (job.workspace != nullptr) {
    return job.workspace->buffers;
  }
  return job.graph_buffers;
}

// Physical address order is used only to group equal owners. The minimum
// declaration ordinal in each group retains the existing attribution rule.
constexpr std::size_t JobOccurrenceCapacity = PipelineRouteCapacity * 2u;
using JobAttribution = std::bitset<JobOccurrenceCapacity>;
static_assert(JobOccurrenceCapacity <=
              std::numeric_limits<std::uint16_t>::max());

[[nodiscard]] const JobState *job_at(const PipelineState &state,
                                     const std::size_t ordinal) noexcept {
  const PipelineStep &step = state.steps[ordinal / 2u];
  return ordinal % 2u == 0u ? step.job.get() : step.alternate_job.get();
}

[[nodiscard]] JobAttribution
first_job_occurrences(const PipelineState &state) noexcept {
  JobAttribution first{};
  std::array<std::uint16_t, JobOccurrenceCapacity> order;
  std::size_t count = 0u;
  const JobState *previous = nullptr;
  for (std::size_t ordinal = 0u; ordinal < state.steps.size() * 2u; ++ordinal) {
    const JobState *const job = job_at(state, ordinal);
    if (job != nullptr && job != previous) {
      order[count++] = static_cast<std::uint16_t>(ordinal);
      previous = job;
    }
  }
  const auto active = std::span{order}.first(count);
  std::sort(active.begin(), active.end(),
            [&state](const std::uint16_t left, const std::uint16_t right) {
              const JobState *const a = job_at(state, left);
              const JobState *const b = job_at(state, right);
              return a == b ? left < right
                            : std::less<const JobState *>{}(a, b);
            });
  previous = nullptr;
  for (const std::uint16_t ordinal : active) {
    const JobState *const job = job_at(state, ordinal);
    if (job != previous) {
      first.set(ordinal);
      previous = job;
    }
  }
  return first;
}

} // namespace

void seal_pipeline_jobs(PipelineState &state) {
  auto &owners = state.memory_inventory.jobs;
  owners.clear();
  const JobAttribution first_jobs = first_job_occurrences(state);
  owners.reserve(first_jobs.count());
  std::array<const JobWorkspace *, PipelineRouteCapacity> attributed_workspaces;
  std::size_t attributed_workspace_count = 0u;
  for (std::size_t ordinal = 0u; ordinal < state.steps.size() * 2u; ++ordinal) {
    if (!first_jobs.test(ordinal)) {
      continue;
    }
    const JobState *const job = job_at(state, ordinal);
    std::uint64_t owned_metadata{};
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
    if (job->workspace != nullptr) {
      const JobWorkspace *const workspace = job->workspace.get();
      const auto end =
          attributed_workspaces.begin() + attributed_workspace_count;
      if (std::find(attributed_workspaces.begin(), end, workspace) == end) {
        if (attributed_workspace_count == attributed_workspaces.size()) {
          owned_metadata = std::numeric_limits<std::uint64_t>::max();
        } else {
          attributed_workspaces[attributed_workspace_count++] = workspace;
        }
        const bool arena_workspace =
            workspace->buffers.borrowed() && workspace->offsets.borrowed();
        if (!arena_workspace) {
          ::rund::detail::counter::Accumulate(owned_metadata,
                                              sizeof(JobWorkspace));
        }
        ::rund::detail::counter::Accumulate(owned_metadata,
                                            vector_memory(workspace->buffers));
        ::rund::detail::counter::Accumulate(owned_metadata,
                                            vector_memory(workspace->offsets));
      }
    }
    std::uint32_t storage_index = ~std::uint32_t{0u};
    if (job->cpu != nullptr && job->cpu->graph != nullptr) {
      const auto found =
          std::find(state.cpu_storage.begin(), state.cpu_storage.end(),
                    job->cpu->graph->storage);
      if (found != state.cpu_storage.end()) {
        storage_index =
            static_cast<std::uint32_t>(found - state.cpu_storage.begin());
      }
    }
    owners.push_back({.ordinal = static_cast<std::uint32_t>(ordinal),
                      .cpu_storage = storage_index,
                      .metadata = owned_metadata});
  }
}

PipelineJobMemory
measure_pipeline_jobs(const PipelineState &state,
                      const PipelineSharedMemory &shared,
                      const std::span<PipelineStepProfile> profiles,
                      const bool include_residency_pool_buffers) noexcept {
  const auto include_buffer =
      [&](const std::shared_ptr<BufferState> &buffer) -> bool {
    return include_residency_pool_buffers ||
           !pipeline_pool_owns_buffer(state.residency_pool.get(), buffer);
  };
  MemoryStats memory = shared.stats;
  const auto &owners = state.memory_inventory.jobs;
  std::size_t owner_index = 0u;
  std::uint64_t metadata = shared.metadata;
  const MemoryStats empty{.backend = state.device->backend,
                          .scope = MemoryScope::Pipeline};
  for (auto &profile :
       profiles.first(std::min(profiles.size(), state.steps.size()))) {
    profile.memory = empty;
  }
  while (owner_index < owners.size()) {
    const std::size_t index = owners[owner_index].ordinal / 2u;
    const PipelineStep &step = state.steps[index];
    MemoryStats owned = empty;
    std::uint64_t owned_metadata{};
    MemoryCounter owned_internal_host{};
    const auto measure_job = [&](const std::shared_ptr<JobState> &job,
                                 const std::size_t ordinal) {
      if (owner_index == owners.size() ||
          owners[owner_index].ordinal != ordinal) {
        return;
      }
      ::rund::detail::counter::Accumulate(owned_metadata,
                                          owners[owner_index++].metadata);
      const bool measure_internal = job->workspace == nullptr;
      BufferMemory internal{};
      if (measure_internal) {
        for (const std::shared_ptr<BufferState> &buffer :
             internal_buffers(*job)) {
          if (include_buffer(buffer)) {
            add_buffer_memory(internal, measure_buffer(buffer));
          }
        }
      }
      BufferMemory view_buffers{};
      const auto measure_view_buffers = [&](const auto &transfers,
                                            const auto &owners) noexcept {
        for (const CpuViewTransfer &transfer : transfers) {
          if (transfer.binding < owners.size()) {
            const std::shared_ptr<BufferState> &buffer =
                owners[transfer.binding];
            if (include_buffer(buffer)) {
              add_buffer_memory(view_buffers, measure_buffer(buffer));
            }
          }
        }
      };
      measure_view_buffers(job->cpu_view_inputs, job->inputs);
      measure_view_buffers(job->cpu_view_outputs, job->outputs);
      merge_memory(owned.resident, fixed_memory(internal.resident));
      merge_memory(owned.resident, fixed_memory(view_buffers.resident));
      if (state.device->backend == Backend::Cpu) {
        merge_memory(owned_internal_host,
                     fixed_memory(internal.physical, internal.reused));
        merge_memory(owned_internal_host,
                     fixed_memory(view_buffers.physical, view_buffers.reused));
      } else {
        merge_memory(owned.device,
                     fixed_memory(internal.physical, internal.reused));
        if (state.device->ops != nullptr &&
            state.device->ops->job_staging != nullptr) {
          merge_memory(owned.staging, state.device->ops->job_staging(*job));
        }
      }
    };
    measure_job(step.job, index * 2u);
    measure_job(step.alternate_job, index * 2u + 1u);
    owned.host = fixed_memory(owned_metadata);
    merge_memory(owned.host, owned_internal_host);
    ::rund::detail::counter::Accumulate(metadata, owned_metadata);
    merge_memory(memory, owned);
    if (index < profiles.size()) {
      profiles[index].memory = owned;
    }
  }
  return PipelineJobMemory{.summary = memory, .metadata = metadata};
}

} // namespace rund::compute::detail

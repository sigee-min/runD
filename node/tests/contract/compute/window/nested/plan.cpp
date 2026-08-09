#include "../../pipeline/local.hpp"
#include "../local.hpp"
#include "local.hpp"

#include <node/runtime/compute/access.hpp>

#include "src/compute/cpu/prepared.hpp"
#include "src/compute/cpu/run/state.hpp"
#include "src/compute/memory/cpu.hpp"
#include "src/compute/pipeline/state.hpp"

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <limits>
#include <span>
#include <utility>
#include <vector>

namespace rund::node::test_contract::window {
[[nodiscard]] constexpr bool PipelineRoutePhaseProjectionIsCanonical() {
  using rund::compute::PipelineNestedPhase;
  using rund::compute::detail::pipeline_nested_phase;
  using rund::compute::detail::PipelineRoute;
  return pipeline_nested_phase(PipelineRoute::Ordinary) ==
             PipelineNestedPhase::None &&
         pipeline_nested_phase(PipelineRoute::NestedSeed) ==
             PipelineNestedPhase::Seed &&
         pipeline_nested_phase(PipelineRoute::NestedAction) ==
             PipelineNestedPhase::Action &&
         pipeline_nested_phase(PipelineRoute::NestedFold) ==
             PipelineNestedPhase::Fold &&
         pipeline_nested_phase(static_cast<PipelineRoute>(0xffu)) ==
             PipelineNestedPhase::None;
}

static_assert(PipelineRoutePhaseProjectionIsCanonical());

[[nodiscard]] bool
CpuRouteOwnershipIsExact(const rund::compute::Pipeline &pipeline) {
  using namespace rund::compute;
  using namespace rund::compute::detail;
  const std::shared_ptr<PipelineState> &state =
      PipelineStateAccess::state(pipeline);
  if (state == nullptr || state->device == nullptr ||
      state->device->backend != Backend::Cpu || state->cpu_storage.empty() ||
      state->cpu_storage.size() > kTemplates) {
    return false;
  }
  for (std::size_t index = 0u; index < state->cpu_storage.size(); ++index) {
    const std::shared_ptr<CpuGraphStorage> &storage = state->cpu_storage[index];
    if (storage == nullptr || storage->program == nullptr) {
      return false;
    }
    for (std::size_t prior = 0u; prior < index; ++prior) {
      if (state->cpu_storage[prior].get() == storage.get() ||
          state->cpu_storage[prior]->program == storage->program) {
        return false;
      }
    }
  }

  std::array<const JobState *, 2u * kTemplates> jobs{};
  std::array<bool, kTemplates> storage_used{};
  std::size_t job_count = 0u;
  bool shared_storage_with_independent_routes = false;
  const auto inspect = [&](const std::shared_ptr<JobState> &job) {
    if (job == nullptr) {
      return false;
    }
    if (std::find(jobs.begin(), jobs.begin() + job_count, job.get()) !=
        jobs.begin() + job_count) {
      return true;
    }
    if (job_count == jobs.size() || job->program == nullptr ||
        job->program->cpu_graph == nullptr || job->cpu == nullptr ||
        job->cpu->graph == nullptr || job->cpu->graph->storage == nullptr) {
      return false;
    }
    const auto planned = plan_cpu_run_route(job->program);
    if (!planned) {
      return false;
    }
    CpuGraphRun &route = *job->cpu->graph;
    std::size_t storage_index = state->cpu_storage.size();
    for (std::size_t index = 0u; index < state->cpu_storage.size(); ++index) {
      if (state->cpu_storage[index].get() == route.storage.get()) {
        storage_index = index;
        break;
      }
    }
    if (storage_index == state->cpu_storage.size() ||
        route.storage->program != job->program->cpu_graph.get()) {
      return false;
    }
    storage_used[storage_index] = true;

    for (std::size_t prior = 0u; prior < job_count; ++prior) {
      const JobState &other = *jobs[prior];
      if (other.cpu.get() == job->cpu.get() ||
          other.cpu->graph.get() == job->cpu->graph.get()) {
        return false;
      }
      if (other.program.get() != job->program.get()) {
        continue;
      }
      const CpuGraphRun &other_route = *other.cpu->graph;
      if (other_route.storage.get() != route.storage.get() ||
          (!route.maps.empty() &&
           other_route.maps.data() == route.maps.data()) ||
          (!route.reads.empty() &&
           other_route.reads.data() == route.reads.data()) ||
          (!route.writes.empty() &&
           other_route.writes.data() == route.writes.data())) {
        return false;
      }
      shared_storage_with_independent_routes = true;
    }
    jobs[job_count++] = job.get();
    return true;
  };

  for (const PipelineStep &step : state->steps) {
    if (!inspect(step.job) ||
        (step.alternate_job != nullptr && !inspect(step.alternate_job))) {
      return false;
    }
  }
  return job_count != 0u && shared_storage_with_independent_routes &&
         std::all_of(storage_used.begin(),
                     storage_used.begin() + state->cpu_storage.size(),
                     [](const bool used) { return used; });
}

[[nodiscard]] std::uint32_t SerialOracle(const std::uint32_t count) noexcept {
  std::uint32_t result = kOuterSeed;
  for (std::size_t index = 0u; index < count; ++index) {
    result += kDomainValues[kQueue[index]];
  }
  result += count * static_cast<std::uint32_t>(kInner);
  return result;
}

[[nodiscard]] bool WarmSetupClean(const rund::compute::Stats &stats) noexcept {
  return stats.pipeline_compiles == 0u && stats.buffer_allocations == 0u &&
         stats.pipeline_cache_evictions == 0u &&
         stats.descriptor_pool_creations == 0u &&
         stats.descriptor_set_allocations == 0u && stats.uploaded_bytes == 0u &&
         stats.download_events == 0u && stats.downloaded_bytes == 0u &&
         stats.output_hash == 0u;
}

template <class Seed, class Action, class Fold>
[[nodiscard]] bool PlanShape(const rund::compute::PipelinePlan &plan,
                             const Seed &seed, const Action &action,
                             const Fold &fold) noexcept {
  constexpr std::uint64_t internal_bytes =
      (kOuter + 5u) * sizeof(std::uint32_t);
  const std::uint64_t infrastructure = plan.state_bytes + plan.prepared_bytes;
  const std::uint64_t logical_workspace =
      kOuter * (seed.graph().memory.logical_bytes +
                kInner * action.graph().memory.logical_bytes +
                fold.graph().memory.logical_bytes);
  const std::uint64_t live_workspace = std::max(
      {seed.graph().memory.live_bytes, action.graph().memory.live_bytes,
       fold.graph().memory.live_bytes});
  return plan.outer_window_count == kOuter && plan.tile_capacity == kTile &&
         plan.inner_iteration_count == kInner &&
         plan.prepared_template_count == kPreparedTemplates &&
         plan.prepared_command_count == kCommands &&
         plan.barrier_count == kTemplates - 1u && plan.resource_count == 11u &&
         plan.state_bytes == internal_bytes && plan.publish_count == 1u &&
         plan.publish_bytes == sizeof(std::uint32_t) &&
         plan.logical_bytes == infrastructure + logical_workspace &&
         plan.live_bytes == infrastructure + live_workspace &&
         plan.physical_bytes == infrastructure + plan.transient_bytes &&
         plan.physical_bytes == plan.peak_bytes &&
         plan.total_bytes == plan.persistent_bytes + plan.physical_bytes;
}

[[nodiscard]] bool PreparedShape(const rund::compute::Pipeline &pipeline) {
  using namespace rund::compute::detail;
  const std::shared_ptr<PipelineState> &state =
      PipelineStateAccess::state(pipeline);
  if (state == nullptr || state->steps.size() != kTemplates ||
      state->windows.size() != 1u || state->resources.size() != 11u ||
      state->publications.size() != 1u) {
    return false;
  }
  const PipelineWindow &nested = state->windows[0u];
  const auto &shape = nested.nested_shape;
  if (!nested.nested() || shape.first() != 0u || shape.end() != kTemplates ||
      shape.seed_first() != 0u || shape.seed_count() != kOuter ||
      shape.action_first() != kOuter || shape.action_count() != kInner ||
      shape.fold_first() != kOuter + kInner ||
      nested.control.maximum != kMaximum || nested.control.tile != kTile) {
    return false;
  }
  for (std::size_t index = 0u; index < state->steps.size(); ++index) {
    const PipelineRoute expected =
        index < kOuter ? PipelineRoute::NestedSeed
                       : (index < kOuter + kInner ? PipelineRoute::NestedAction
                                                  : PipelineRoute::NestedFold);
    if (state->steps[index].route != expected ||
        state->steps[index].job == nullptr) {
      return false;
    }
  }

  std::array<const JobState *, kTemplates> owners{};
  std::size_t owner_count = 0u;
  for (const PipelineStep &step : state->steps) {
    const JobState *const owner = step.job.get();
    const auto found =
        std::find(owners.begin(), owners.begin() + owner_count, owner);
    if (found == owners.begin() + owner_count) {
      owners[owner_count++] = owner;
    }
  }
  constexpr std::size_t expected_owners =
      kOuter + (kInner == 1u ? 1u : 2u) + 3u;
  return owner_count == expected_owners &&
         state->steps[kOuter].job == state->steps[kOuter + 2u].job;
}

[[nodiscard]] std::size_t
ActionOwnerCount(const rund::compute::Pipeline &pipeline,
                 const rund::compute::graph::Fingerprint action) {
  using namespace rund::compute::detail;
  const std::shared_ptr<PipelineState> &state =
      PipelineStateAccess::state(pipeline);
  if (state == nullptr || state->windows.size() != 1u) {
    return std::numeric_limits<std::size_t>::max();
  }
  const PipelineWindow &nested = state->windows[0u];
  std::array<const JobState *, 2u> owners{};
  std::size_t owner_count = 0u;
  for (std::size_t index = nested.nested_shape.action_first();
       index < nested.nested_shape.fold_first(); ++index) {
    if (index >= state->steps.size() ||
        state->steps[index].program == nullptr ||
        state->steps[index].job == nullptr ||
        state->steps[index].program->graph_info.fingerprint != action) {
      return std::numeric_limits<std::size_t>::max();
    }
    const JobState *const owner = state->steps[index].job.get();
    const auto found =
        std::find(owners.begin(), owners.begin() + owner_count, owner);
    if (found == owners.begin() + owner_count) {
      if (owner_count == owners.size()) {
        return std::numeric_limits<std::size_t>::max();
      }
      owners[owner_count++] = owner;
    }
  }
  return owner_count;
}

// Test-only structural oracle. Warm production execution never performs this
// compact owner walk; the snapshot directly proves retained owner stability
// without adding an O(K + N) scan to the runtime path.

bool NestedPlanShape(const rund::compute::PipelinePlan &plan,
                     const NestedSeed &seed, const NestedAction &action,
                     const NestedFold &fold) {
  return PlanShape(plan, seed, action, fold);
}

} // namespace rund::node::test_contract::window

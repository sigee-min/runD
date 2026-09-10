#include "src/compute/pipeline/state.hpp"

namespace rund::compute::detail {
struct PipelineMemoryPlan;
struct PipelineBuildState;
}
namespace {
template <class T> concept CompletePipelineType = requires { sizeof(T); };
static_assert(!CompletePipelineType<rund::compute::detail::PipelineMemoryPlan>);
static_assert(!CompletePipelineType<rund::compute::detail::PipelineBuildState>);
} // namespace

#include "local.hpp"

#include "../../allocation.hpp"
#include "src/compute/memory/local.hpp"
#include "src/compute/backend.hpp"
#include "src/compute/pipeline/run/memory/internal.hpp"

#include <array>
#include <cstddef>
#include <memory>
#include <limits>
#include <vector>

namespace rund_node_test_pipeline::memory {

int CheckAttribution() {
  using namespace rund::compute;
  using namespace rund::compute::detail;
  PipelineState state{};
  state.device = std::make_shared<DeviceState>();
  state.device->backend = Backend::Cpu;
  auto workspace = std::make_shared<JobWorkspace>();
  std::array<std::shared_ptr<JobState>, 4u> jobs;
  for (auto &job : jobs) {
    job = std::make_shared<JobState>();
    job->workspace = workspace;
  }
  // Address order cannot decide attribution: alternate-only, null, repeated,
  // and nonadjacent shared owners belong to their first declared occurrence.
  state.steps = {{.alternate_job = jobs[3]},
                 {.job = jobs[1], .alternate_job = jobs[3]},
                 {.job = jobs[1]},
                 {.job = jobs[2], .alternate_job = jobs[0]},
                 {.job = jobs[3], .alternate_job = jobs[2]}};
  const std::array<std::size_t, 5u> first_counts{1u, 1u, 0u, 2u, 0u};
  std::vector<PipelineStepProfile> profiles(state.steps.size());
  seal_pipeline_jobs(state);
  seal_pipeline_shared(state);
  node_compute_allocation::Start();
  const auto observed = measure_pipeline_jobs(state, {}, profiles, true);
  node_compute_allocation::Stop();
  if (node_compute_allocation::Count() != 0u ||
      observed.metadata != 4u * sizeof(JobState) + sizeof(JobWorkspace)) {
    return 31;
  }
  for (std::size_t index = 0u; index < profiles.size(); ++index) {
    const auto expected = first_counts[index] * sizeof(JobState) +
                          (index == 0u ? sizeof(JobWorkspace) : 0u);
    if (profiles[index].memory.host != fixed_memory(expected)) {
      return 32;
    }
  }
  // A repeated/null row must erase stale attribution, including when only
  // a prefix of profiles is requested. Adding no counters preserves even
  // saturated shared totals and their nonzero budgets.
  state.steps = {{.job = jobs[0]}, {}, {.alternate_job = jobs[0]}};
  profiles.resize(2u);
  PipelineSharedMemory shared{};
  shared.stats.backend = Backend::Cpu;
  shared.stats.scope = MemoryScope::Pipeline;
  shared.stats.host = fixed_memory(std::numeric_limits<std::uint64_t>::max());
  shared.stats.host.budget = 19u;
  shared.metadata = std::numeric_limits<std::uint64_t>::max();
  for (auto &profile : profiles) {
    profile.memory = shared.stats;
  }
  seal_pipeline_jobs(state);
  seal_pipeline_shared(state);
  node_compute_allocation::Start();
  const auto repeated = measure_pipeline_jobs(state, shared, profiles, true);
  node_compute_allocation::Stop();
  const MemoryStats empty_row{.backend = Backend::Cpu,
                              .scope = MemoryScope::Pipeline};
  auto expected_repeated = shared.stats;
  merge_memory(expected_repeated.host,
               fixed_memory(sizeof(JobState) + sizeof(JobWorkspace)));
  if (node_compute_allocation::Count() != 0u ||
      profiles[1].memory != empty_row || repeated.summary != expected_repeated ||
      repeated.metadata != shared.metadata) {
    return 39;
  }
  profiles.resize(3u);
  profiles[2].memory = shared.stats;
  const auto complete = measure_pipeline_jobs(state, shared, profiles, true);
  if (profiles[2].memory != empty_row || complete.summary != repeated.summary) {
    return 40;
  }
  // The maximum primary/alternate occurrence domain must fit the compact
  // scratch ordinals without truncation, loss, or heap allocation.
  state.steps.clear();
  state.steps.resize(PipelineRouteCapacity);
  for (auto &step : state.steps) {
    step.job = std::make_shared<JobState>();
    step.alternate_job = std::make_shared<JobState>();
    step.job->workspace = workspace;
    step.alternate_job->workspace = workspace;
  }
  profiles.resize(state.steps.size());
  seal_pipeline_jobs(state);
  seal_pipeline_shared(state);
  node_compute_allocation::Start();
  const auto maximum = measure_pipeline_jobs(state, {}, profiles, true);
  node_compute_allocation::Stop();
  if (node_compute_allocation::Count() != 0u ||
      maximum.metadata != 2u * PipelineRouteCapacity * sizeof(JobState) +
                              sizeof(JobWorkspace)) {
    return 33;
  }
  for (std::size_t index = 0u; index < profiles.size(); ++index) {
    const auto expected = 2u * sizeof(JobState) +
                          (index == 0u ? sizeof(JobWorkspace) : 0u);
    if (profiles[index].memory.host != fixed_memory(expected)) {
      return 34;
    }
  }
  // No profile destination still produces the identical aggregate.
  const auto summary_only = measure_pipeline_jobs(state, {}, {}, true);
  return summary_only.metadata == maximum.metadata &&
                 summary_only.summary == maximum.summary
             ? 0
             : 35;
}

int CheckSharedAttribution() {
  using namespace rund::compute;
  using namespace rund::compute::detail;
  PipelineState state{};
  // This tests the backend-neutral host observer; no DeviceOps or native
  // objects are involved in the fixture.
  state.device = std::make_shared<DeviceState>();
  state.device->backend = Backend::Metal;
  std::array<std::shared_ptr<JobWorkspace>, 3u> workspaces;
  std::array<std::shared_ptr<JobState>, 3u> jobs;
  for (std::size_t i = 0u; i < jobs.size(); ++i) {
    workspaces[i] = std::make_shared<JobWorkspace>();
    jobs[i] = std::make_shared<JobState>();
    jobs[i]->workspace = workspaces[i];
  }
  state.steps.resize(PipelineIterationCapacity);
  for (std::size_t i = 0u; i < state.steps.size(); ++i) {
    state.steps[i].job = jobs[i % jobs.size()];
    state.steps[i].alternate_job = jobs[(i + 1u) % jobs.size()];
  }
  seal_pipeline_jobs(state);
  seal_pipeline_shared(state);
  const auto empty = measure_pipeline_shared(state, true);
  auto arena = std::make_shared<JobArena>();
  for (auto &workspace : workspaces) {
    workspace->arena = arena;
  }
  seal_pipeline_jobs(state);
  seal_pipeline_shared(state);
  node_compute_allocation::Start();
  const auto shared = measure_pipeline_shared(state, true);
  node_compute_allocation::Stop();
  if (node_compute_allocation::Count() != 0u ||
      shared.metadata != empty.metadata + sizeof(JobArena)) {
    return 36;
  }
  auto expected = empty.stats;
  merge_memory(expected.host, fixed_memory(sizeof(JobArena)));
  if (shared.stats != expected) {
    return 37;
  }
  workspaces[2]->arena = std::make_shared<JobArena>();
  seal_pipeline_shared(state);
  const auto separate = measure_pipeline_shared(state, true);
  merge_memory(expected.host, fixed_memory(sizeof(JobArena)));
  if (separate.metadata != empty.metadata + 2u * sizeof(JobArena) ||
      separate.stats != expected) {
    return 38;
  }
  // Frozen ownership must not freeze mutable observation coordinates.
  DeviceOps ops{};
  ops.pipeline_memory = [](const PipelineState &value) noexcept {
    rund::node::accel::detail::PreparedPipelineMemory result{};
    result.host.current = value.transfer_bytes;
    result.host.peak = value.transfer_bytes;
    result.host.cumulative = value.transfer_bytes;
    return result;
  };
  state.device->ops = &ops;
  for (const std::uint64_t epoch : {17u, 91u}) {
    state.transfer_bytes = epoch;
    state.transfer_peak = epoch;
    state.frame_current = epoch;
    state.frame_peak = epoch;
    state.frame_bytes = epoch;
    node_compute_allocation::Start();
    const auto live = measure_pipeline_shared(state, true);
    node_compute_allocation::Stop();
    if (node_compute_allocation::Count() != 0u ||
        live.metadata != separate.metadata ||
        live.stats.transfer.cumulative != epoch ||
        live.stats.frame.current != epoch ||
        live.stats.host.current != separate.stats.host.current + epoch ||
        live.prepared.host.current != epoch) {
      return 44;
    }
  }
  state.device->ops = nullptr;
  return 0;
}

} // namespace rund_node_test_pipeline::memory

#include "local.hpp"

#include <node/runtime/compute/access.hpp>

#include "src/compute/memory/arena.hpp"
#include "src/compute/pipeline/plan/arena.hpp"
#include "src/compute/pipeline/plan/local.hpp"
#include "src/compute/pipeline/state.hpp"
#include "src/compute/pipeline/state/assembly.hpp"
#include "src/compute/status.hpp"

#include <array>
#include <cstdio>
#include <limits>
#include <memory>
#include <vector>

namespace rund_node_test_pipeline::memory {

[[nodiscard]] int CheckArena(rund::compute::Device &device) {
  using namespace rund::compute;
  const bool accelerated =
      detail::DeviceAccess::state(device)->backend != Backend::Cpu;
  constexpr std::array<std::size_t, 2u> arena_chunks{9u, 5u};
  auto arena_program = MakeProgram(device, arena_chunks);
  detail::PipelineBuildState arena_build{};
  arena_build.device = detail::DeviceAccess::state(device);
  detail::PipelineBuildStep arena_step{};
  arena_step.program = arena_program;
  arena_step.logical_step = 0u;
  arena_build.steps.push_back(std::move(arena_step));

  // Large repeated routes retain one parallel entry per logical chunk,
  // with exact prefix boundaries and the same placement as a single step.
  std::vector<detail::PipelineBuildStep> repeated(128u);
  for (auto &step : repeated) {
    step.program = arena_program;
  }
  detail::PipelineMemoryPlan single_plan{};
  detail::PipelineMemoryPlan repeated_plan{};
  if (!detail::plan_pipeline_arena(*arena_build.device,
                                   std::span{repeated}.first(1u), single_plan) ||
      !detail::plan_pipeline_arena(*arena_build.device, repeated, repeated_plan) ||
      repeated_plan.steps.size() != repeated.size() + 1u ||
      repeated_plan.offsets.size() != repeated.size() * arena_chunks.size() ||
      repeated_plan.owners.size() != repeated_plan.offsets.size() ||
      repeated_plan.chunks != single_plan.chunks) {
    return 31;
  }
  for (std::size_t i = 0u; i < repeated.size(); ++i) {
    if (repeated_plan.steps[i] != i * arena_chunks.size()) {
      return 32;
    }
    for (std::size_t chunk = 0u; chunk < arena_chunks.size(); ++chunk) {
      const std::size_t row = repeated_plan.steps[i] + chunk;
      if (repeated_plan.offsets[row] != single_plan.offsets[chunk] ||
          repeated_plan.owners[row] != single_plan.owners[chunk]) {
        return 33;
      }
    }
  }
  if (repeated_plan.steps.back() != repeated_plan.offsets.size()) {
    return 34;
  }

  // The arena planner receives a compact route, so its public largest/peak
  // phase must consume the common route projector without inventing an outer
  // occurrence for the reusable Fold route.
  std::array<detail::PipelineBuildStep, 1u> fold_steps{};
  fold_steps.front().program = arena_program;
  fold_steps.front().logical_step = 7u;
  fold_steps.front().iteration = 2u;
  fold_steps.front().route = detail::PipelineRoute::NestedFold;
  detail::PipelineMemoryPlan fold_plan{};
  const Status fold_planned = detail::plan_pipeline_arena(
      *arena_build.device,
      std::span<const detail::PipelineBuildStep>{fold_steps}, fold_plan);
  if (!fold_planned ||
      fold_plan.summary.largest_nested_phase != PipelineNestedPhase::Fold ||
      fold_plan.summary.peak_nested_phase != PipelineNestedPhase::Fold ||
      fold_plan.summary.largest_step != 7u ||
      fold_plan.summary.largest_iteration != 2u ||
      fold_plan.summary.peak_step != 7u ||
      fold_plan.summary.peak_iteration != 2u ||
      fold_plan.summary.largest_outer_window !=
          std::numeric_limits<std::size_t>::max() ||
      fold_plan.summary.largest_inner_iteration !=
          std::numeric_limits<std::size_t>::max() ||
      fold_plan.summary.peak_outer_window !=
          std::numeric_limits<std::size_t>::max() ||
      fold_plan.summary.peak_inner_iteration !=
          std::numeric_limits<std::size_t>::max()) {
    return 25;
  }

  const auto arena_plan = detail::plan_memory(arena_build);
  if (!arena_plan || (*arena_plan)->frozen == nullptr ||
      (*arena_plan)->summary.allocation_count != 1u ||
      (*arena_plan)->summary.transient_bytes != 69u * sizeof(std::uint32_t) ||
      (*arena_plan)->steps != std::vector<std::size_t>{0u, 2u} ||
      (*arena_plan)->owners != std::vector<std::size_t>{0u, 0u} ||
      (*arena_plan)->offsets != std::vector<std::size_t>{0u, 64u} ||
      (*arena_plan)->chunks != std::vector<std::size_t>{69u}) {
    if (arena_plan) {
      std::fprintf(stderr,
                   "arena plan alloc=%llu transient=%llu steps=%zu owners=%zu "
                   "offsets=%zu chunks=%zu\n",
                   static_cast<unsigned long long>(
                       (*arena_plan)->summary.allocation_count),
                   static_cast<unsigned long long>(
                       (*arena_plan)->summary.transient_bytes),
                   (*arena_plan)->steps.size(), (*arena_plan)->owners.size(),
                   (*arena_plan)->offsets.size(), (*arena_plan)->chunks.size());
    } else {
      std::fprintf(stderr, "arena plan rejected reason=%u\n",
                   static_cast<unsigned>(arena_plan.reason()));
    }
    return 11;
  }
  const auto arena_memory = detail::make_pipeline_memory(
      arena_build.device, (*arena_plan)->frozen->steps, **arena_plan);
  if (!arena_memory || arena_memory->buffers.size() != 1u ||
      arena_memory->steps.size() != 1u ||
      arena_memory->steps.front() == nullptr ||
      arena_memory->steps.front()->buffers.size() != 2u ||
      arena_memory->steps.front()->offsets.size() != 2u ||
      arena_memory->steps.front()->offsets[0u] != 0u ||
      arena_memory->steps.front()->offsets[1u] != 64u ||
      (!accelerated && (!arena_memory->steps.front()->buffers.borrowed() ||
                        !arena_memory->steps.front()->offsets.borrowed() ||
                        arena_memory->cpu_prepared_arena == nullptr ||
                        arena_memory->steps.front().owner_before(
                            arena_memory->cpu_prepared_arena) ||
                        arena_memory->cpu_prepared_arena.owner_before(
                            arena_memory->steps.front()))) ||
      (accelerated && (arena_memory->steps.front()->buffers.borrowed() ||
                       arena_memory->steps.front()->offsets.borrowed() ||
                       arena_memory->cpu_prepared_arena != nullptr)) ||
      arena_memory->steps.front()->buffers[0u] !=
          arena_memory->buffers.front() ||
      arena_memory->steps.front()->buffers[1u] !=
          arena_memory->buffers.front()) {
    if (!arena_memory) {
      std::fprintf(stderr, "arena materialization rejected reason=%u\n",
                   static_cast<unsigned>(arena_memory.reason()));
    } else {
      const auto &workspace = arena_memory->steps.front();
      std::fprintf(
          stderr,
          "arena workspace buffers=%zu/%d offsets=%zu/%d prepared=%d "
          "shared=%d values=%zu,%zu owners=%d,%d\n",
          workspace == nullptr ? 0u : workspace->buffers.size(),
          workspace != nullptr && workspace->buffers.borrowed(),
          workspace == nullptr ? 0u : workspace->offsets.size(),
          workspace != nullptr && workspace->offsets.borrowed(),
          arena_memory->cpu_prepared_arena != nullptr,
          workspace != nullptr &&
              !workspace.owner_before(arena_memory->cpu_prepared_arena) &&
              !arena_memory->cpu_prepared_arena.owner_before(workspace),
          workspace == nullptr || workspace->offsets.size() < 1u
              ? std::numeric_limits<std::size_t>::max()
              : workspace->offsets[0u],
          workspace == nullptr || workspace->offsets.size() < 2u
              ? std::numeric_limits<std::size_t>::max()
              : workspace->offsets[1u],
          workspace != nullptr && workspace->buffers.size() > 0u &&
              workspace->buffers[0u] == arena_memory->buffers.front(),
          workspace != nullptr && workspace->buffers.size() > 1u &&
              workspace->buffers[1u] == arena_memory->buffers.front());
    }
    return 12;
  }

  // A present recurrence follower cannot copy a null materialized owner and
  // fall through to private Job allocation. Corrupt only the owner's sealed
  // presence record; the follower must reject instead of becoming null.
  detail::PipelineBuildState follower_build{};
  follower_build.device = arena_build.device;
  for (std::size_t iteration = 0u; iteration < 2u; ++iteration) {
    detail::PipelineBuildStep follower{};
    follower.program = arena_program;
    follower.logical_step = 3u;
    follower.iteration = iteration;
    follower.iteration_bound = 2u;
    follower_build.steps.push_back(std::move(follower));
  }
  const auto follower_plan = detail::plan_memory(follower_build);
  if (!follower_plan || (*follower_plan)->frozen == nullptr ||
      (*follower_plan)->workspace_routes.size() != 2u ||
      !(*follower_plan)->workspace_routes[0u].owns(0u) ||
      (*follower_plan)->workspace_routes[1u].owner != 0u) {
    return 26;
  }
  const auto follower_memory = detail::make_pipeline_memory(
      follower_build.device, (*follower_plan)->frozen->steps, **follower_plan);
  if (!follower_memory || follower_memory->steps.size() != 2u ||
      follower_memory->steps[0u] == nullptr ||
      follower_memory->steps[0u] != follower_memory->steps[1u] ||
      follower_memory->steps[0u]->arena != nullptr) {
    return 26;
  }
  detail::PipelineMemoryPlan missing_owner = **follower_plan;
  missing_owner.workspace_routes[0u] = {};
  const auto rejected_follower = detail::make_pipeline_memory(
      follower_build.device, missing_owner.frozen->steps, missing_owner);
  if (rejected_follower ||
      rejected_follower.reason() != Reason::PipelineInvalid) {
    return 26;
  }
  return 0;
}

} // namespace rund_node_test_pipeline::memory

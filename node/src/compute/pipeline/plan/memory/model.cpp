#include "local.hpp"

#include "../arena.hpp"
#include "../compare.hpp"
#include "../prepare.hpp"
#include "../resource.hpp"

#include "../../../../accel/kernel/recurrence.hpp"
#include "../../../backend.hpp"
#include "../../../buffer/local.hpp"
#include "../../../cpu/run/state.hpp"
#include "../../../job/local.hpp"
#include "../../../memory/arena.hpp"
#include "../../../status.hpp"
#include "../../../type.hpp"

#include <kernel/core/checked.hpp>
#include <rund/compute/abi/observe.hpp>

#include <algorithm>
#include <cstdint>
#include <limits>
#include <memory>
#include <new>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

namespace rund::compute::detail {

[[nodiscard]] Result<PipelinePlanningModel>
initialize_pipeline_plan(const PipelineBuildState &build,
                         PipelineMemoryPlan &plan) {
  PipelinePlan &summary = plan.summary;
  std::vector<PipelineFrozenStep> frozen_steps;
  frozen_steps.reserve(build.steps.size());
  for (const PipelineBuildStep &step : build.steps) {
    frozen_steps.push_back(PipelineFrozenStep{
        .program = step.program,
        .logical_step = step.logical_step,
        .iteration = step.iteration,
        .iteration_bound = step.iteration_bound,
        .nested = step.nested,
        .route = step.route,
        .writes_each_iteration = step.writes_each_iteration,
    });
  }
  std::vector<PipelineFrozenNestedWindow> frozen_nested;
  frozen_nested.reserve(build.nested_windows.size());
  for (const PipelineBuildNestedWindow &nested : build.nested_windows) {
    frozen_nested.push_back(PipelineFrozenNestedWindow{
        .shape = nested.shape,
        .recurrent_output_count = nested.recurrent_output_count,
    });
  }
  plan.frozen =
      std::make_shared<const PipelineBuildSnapshot>(PipelineBuildSnapshot{
          .device = build.device,
          .steps = std::move(frozen_steps),
          .nested_windows = std::move(frozen_nested),
          .logical_step_count = build.logical_step_count,
          .sealed_repetitions = build.sealed_repetitions,
          .profile = build.profile,
          .commit = build.commit,
      });
  std::unordered_set<const ProgramState *> programs;
  programs.reserve(build.steps.size());
  std::vector<const ProgramState *> unique_programs;
  unique_programs.reserve(build.steps.size());
  std::unordered_map<const ProgramState *, std::size_t> cpu_programs;
  cpu_programs.reserve(build.steps.size());
  plan.cpu_storage_by_step.assign(build.steps.size(),
                                  std::numeric_limits<std::size_t>::max());
  plan.cpu_route_slices.resize(build.steps.size());
  if (!build.state_pairs.empty()) {
    plan.cpu_alternate_route_slices.resize(build.steps.size());
  }
  plan.job_owners.resize(build.steps.size());
  for (std::size_t index = 0u; index < build.steps.size(); ++index) {
    const PipelineBuildStep &step = build.steps[index];
    if (step.program == nullptr) {
      return Result<PipelinePlanningModel>::fail(Reason::PipelineInvalid);
    }
    if (programs.insert(step.program.get()).second) {
      unique_programs.push_back(step.program.get());
      if (!kernel::checked::add(
              summary.node_count,
              static_cast<std::uint64_t>(step.program->graph_info.nodes.size()),
              summary.node_count)) {
        return Result<PipelinePlanningModel>::fail(Reason::PipelineCapacity);
      }
    }
    if (build.device->backend == Backend::Cpu && !step.program->empty() &&
        step.program->cpu_graph != nullptr) {
      const auto [position, inserted] =
          cpu_programs.emplace(step.program.get(), plan.cpu_programs.size());
      if (inserted) {
        const auto storage = plan_cpu_graph_storage(step.program);
        const auto route = plan_cpu_run_route(step.program);
        if (!storage || !route) {
          return Result<PipelinePlanningModel>::fail(
              storage ? route.reason() : storage.reason());
        }
        plan.cpu_programs.push_back(step.program);
        plan.cpu_storage_plans.push_back(*storage);
        plan.cpu_route_plans.push_back(*route);
        if (!merge_cpu_execution_storage_plan(plan.cpu_prepared_arena.execution,
                                              storage->execution)) {
          return Result<PipelinePlanningModel>::fail(Reason::PipelineCapacity);
        }
        if (!kernel::checked::add(summary.prepared_host_bytes,
                                  storage->private_total.host,
                                  summary.prepared_host_bytes) ||
            !kernel::checked::add(summary.prepared_tile_bytes,
                                  storage->private_total.tile,
                                  summary.prepared_tile_bytes)) {
          return Result<PipelinePlanningModel>::fail(Reason::PipelineCapacity);
        }
      }
      plan.cpu_storage_by_step[index] = position->second;
    }
  }

  // Seal resource Views, state ordinals, and Window controls before any
  // private-Job reuse or accounting decision. All post-boundary consumers
  // below read this plan instead of interpreting authored bindings/scalars.
  auto schedule = plan_pipeline_schedule(build, plan);
  if (!schedule) {
    return Result<PipelinePlanningModel>::fail(schedule.reason(),
                                               schedule.location());
  }
  if (plan.step_resources.size() != build.steps.size() ||
      plan.window_states.size() != build.steps.size() ||
      std::any_of(plan.window_states.begin(), plan.window_states.end(),
                  [&](const std::uint32_t state) {
                    return state != PipelineResourceUnassigned &&
                           state >= plan.window_controls.size();
                  })) {
    return Result<PipelinePlanningModel>::fail(Reason::PipelineInvalid);
  }

  for (std::size_t index = 0u; index < build.steps.size(); ++index) {
    const PipelineBuildStep &step = build.steps[index];
    std::size_t owner = index;
    const auto control =
        [&](const std::size_t step_index) -> const PipelineWindowControl * {
      const std::uint32_t state = plan.window_states[step_index];
      return state == PipelineResourceUnassigned ? nullptr
                                                 : &plan.window_controls[state];
    };
    std::size_t candidate = index;
    if (step.nested != 0u) {
      const std::size_t nested_index = step.nested - 1u;
      if (nested_index >= build.nested_windows.size() ||
          !build.nested_windows[nested_index].shape.retained_owner(index,
                                                                   candidate)) {
        return Result<PipelinePlanningModel>::fail(Reason::PipelineInvalid);
      }
    } else if (step.iteration_bound > 1u && step.iteration >= 3u &&
               index >= 2u) {
      candidate = index - 2u;
    }
    if (candidate < index) {
      const bool same = same_recurrence_phase(
          step, plan.step_resources[index], control(index),
          build.steps[candidate], plan.step_resources[candidate],
          control(candidate));
      if (step.nested != 0u && !same) {
        return Result<PipelinePlanningModel>::fail(Reason::PipelineInvalid);
      }
      if (same) {
        owner = plan.job_owners[candidate];
      }
    }
    plan.job_owners[index] = owner;
  }
  const bool transactional = !build.state_pairs.empty();
  for (std::size_t index = 0u; index < build.steps.size(); ++index) {
    if (plan.job_owners[index] != index) {
      continue;
    }
    const std::size_t cpu_owner = plan.cpu_storage_by_step[index];
    if (cpu_owner == std::numeric_limits<std::size_t>::max()) {
      continue;
    }
    if (cpu_owner >= plan.cpu_route_plans.size()) {
      return Result<PipelinePlanningModel>::fail(Reason::PipelineInvalid);
    }
    const CpuRunRoutePlan &route = plan.cpu_route_plans[cpu_owner];
    if (!append_cpu_run_route_slice(plan.cpu_prepared_arena, route,
                                    plan.cpu_route_slices[index]) ||
        (transactional &&
         !append_cpu_run_route_slice(plan.cpu_prepared_arena, route,
                                     plan.cpu_alternate_route_slices[index]))) {
      return Result<PipelinePlanningModel>::fail(Reason::PipelineCapacity);
    }
  }
  return Result<PipelinePlanningModel>::success(
      PipelinePlanningModel{.unique_programs = std::move(unique_programs)});
}

} // namespace rund::compute::detail

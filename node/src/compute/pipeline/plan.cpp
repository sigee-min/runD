#include <rund/compute/pipeline.hpp>
#include <rund/compute/resource/plan.hpp>
#include <rund/counter.hpp>

#include "../backend.hpp"
#include "../buffer/local.hpp"
#include "../job/local.hpp"
#include "../status.hpp"
#include "claim.hpp"
#include "local.hpp"
#include "plan/contract.hpp"
#include "plan/local.hpp"
#include "plan/output.hpp"
#include "plan/prepare.hpp"
#include "state.hpp"

#include <algorithm>
#include <array>
#include <cstdint>
#include <functional>
#include <limits>
#include <memory>
#include <new>
#include <numeric>
#include <span>
#include <unordered_map>
#include <utility>
#include <vector>

namespace rund::compute::detail {
namespace {} // namespace

Result<PipelinePlan>
plan_pipeline(const std::shared_ptr<PipelineBuildState> &build) noexcept {
  if (build == nullptr) {
    return Result<PipelinePlan>::fail(Reason::PipelineInvalid);
  }
  if (build->failure != Reason::Ok) {
    return Result<PipelinePlan>::fail(build->failure);
  }
  if (build->device == nullptr) {
    return Result<PipelinePlan>::fail(Reason::DeviceInvalid);
  }
  if (build->steps.empty()) {
    return Result<PipelinePlan>::fail(Reason::PipelineEmpty);
  }
  if (build->memory == nullptr) {
    auto planned = plan_memory(*build);
    if (!planned) {
      return Result<PipelinePlan>::fail(planned.reason());
    }
    build->memory = std::move(planned).value();
  }
  return Result<PipelinePlan>::success(build->memory->summary);
}

void configure_pipeline_budget(const std::shared_ptr<PipelineBuildState> &build,
                               const MemoryBudget budget) noexcept {
  if (build == nullptr || build->failure != Reason::Ok) {
    return;
  }
  build->budget = budget.bytes;
  build->has_budget = true;
}

PipelinePlan
pipeline_plan(const std::shared_ptr<PipelineState> &state) noexcept {
  return state == nullptr ? PipelinePlan{} : state->plan;
}

Result<std::shared_ptr<PipelineState>>
prepare_pipeline(std::shared_ptr<PipelineBuildState> build) noexcept {
  if (build == nullptr) {
    return Result<std::shared_ptr<PipelineState>>::fail(
        Reason::PipelineInvalid);
  }
  if (build->failure != Reason::Ok) {
    return Result<std::shared_ptr<PipelineState>>::fail(build->failure);
  }
  if (build->device == nullptr) {
    return Result<std::shared_ptr<PipelineState>>::fail(Reason::DeviceInvalid);
  }
  if (build->steps.empty()) {
    return Result<std::shared_ptr<PipelineState>>::fail(Reason::PipelineEmpty);
  }
  const bool nested = !build->nested_windows.empty();
  if (build->steps.size() >
          (nested ? PipelineRouteCapacity : PipelineIterationCapacity) ||
      build->binding_count >
          (nested ? PipelineRouteBindingCapacity : PipelineBindingCapacity) ||
      build->state_pairs.size() > PipelineLeafCapacity ||
      build->publications.size() > PipelineLeafCapacity) {
    return Result<std::shared_ptr<PipelineState>>::fail(
        Reason::PipelineCapacity);
  }
  if (build->commit != !build->state_pairs.empty() ||
      (build->seed != nullptr && !build->commit)) {
    return Result<std::shared_ptr<PipelineState>>::fail(
        Reason::PipelineInvalid);
  }

  auto planned_memory = plan_pipeline(build);
  if (!planned_memory) {
    return Result<std::shared_ptr<PipelineState>>::fail(
        planned_memory.reason());
  }
  if (build->has_budget && planned_memory->peak_bytes > build->budget) {
    return Result<std::shared_ptr<PipelineState>>::fail(
        Reason::PipelineMemoryBudget);
  }
  const Status materialized = materialize_pipeline(*build);
  if (!materialized) {
    return Result<std::shared_ptr<PipelineState>>::fail(materialized.reason());
  }

  try {
    PipelinePrepare preparation{};
    const Status admitted = admit_pipeline(build, preparation);
    if (!admitted) {
      return Result<std::shared_ptr<PipelineState>>::fail(admitted.reason());
    }
    const Status scheduled = schedule_pipeline(build, preparation);
    if (!scheduled) {
      return Result<std::shared_ptr<PipelineState>>::fail(scheduled.reason());
    }
    const Status bound = bind_pipeline(build, preparation);
    if (!bound) {
      return Result<std::shared_ptr<PipelineState>>::fail(
          bound.reason(), preparation.failure);
    }
    std::shared_ptr<PipelineState> &state = preparation.state;
    if (build->seed != nullptr) {
      const Status restored = restore_pipeline_state(state, build->seed);
      if (!restored) {
        return Result<std::shared_ptr<PipelineState>>::fail(restored.reason());
      }
    }
    return Result<std::shared_ptr<PipelineState>>::success(std::move(state));
  } catch (const std::bad_alloc &) {
    return Result<std::shared_ptr<PipelineState>>::fail(
        Reason::PipelineCapacity);
  }
}

} // namespace rund::compute::detail

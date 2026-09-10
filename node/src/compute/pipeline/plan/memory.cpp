#include "../state/assembly.hpp"
#include "memory/local.hpp"

#include <memory>
#include <new>
#include <utility>

namespace rund::compute::detail {

[[nodiscard]] Result<std::shared_ptr<const PipelineMemoryPlan>>
plan_memory(const PipelineBuildState &build) {
  try {
    auto plan = std::make_shared<PipelineMemoryPlan>();
    auto model = initialize_pipeline_plan(build, *plan);
    if (!model) {
      return Result<std::shared_ptr<const PipelineMemoryPlan>>::fail(
          model.reason(), model.location());
    }
    auto workload = project_pipeline_workload(build, *plan);
    if (!workload) {
      return Result<std::shared_ptr<const PipelineMemoryPlan>>::fail(
          workload.reason(), workload.location());
    }
    const Status finalized =
        finalize_pipeline_plan(build, *plan, *model, *workload);
    if (!finalized) {
      return Result<std::shared_ptr<const PipelineMemoryPlan>>::fail(
          finalized.reason());
    }
    return Result<std::shared_ptr<const PipelineMemoryPlan>>::success(
        std::move(plan));
  } catch (const std::bad_alloc &) {
    return Result<std::shared_ptr<const PipelineMemoryPlan>>::fail(
        Reason::PipelineCapacity);
  }
}

} // namespace rund::compute::detail

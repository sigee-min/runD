#include "../../state/assembly.hpp"
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
#include <array>
#include <cstdint>
#include <limits>
#include <memory>
#include <new>
#include <numeric>
#include <optional>
#include <span>
#include <tuple>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

namespace rund::compute::detail {

[[nodiscard]] Status
seal_pipeline_workspace_routes(const PipelineBuildState &build,
                               PipelineMemoryPlan &plan) {
  if (plan.views.size() != build.steps.size()) {
    return Status::fail(Reason::PipelineInvalid);
  }
  plan.workspace_routes.assign(build.steps.size(), {});
  const bool has_job_arena = !plan.view_chunks.empty() || !plan.scratch.empty();
  for (std::size_t index = 0u; index < build.steps.size(); ++index) {
    const PipelineBuildStep &step = build.steps[index];
    if (step.program == nullptr) {
      return Status::fail(Reason::PipelineInvalid);
    }
    // This is the sole workspace-presence decision. CPU dense-View transfer
    // layouts are deliberately absent: they are owned by each private Job.
    const bool present = !step.program->chunks.empty() ||
                         !plan.views[index].empty() ||
                         (has_job_arena && !step.program->empty());
    if (step.iteration_bound > 1u && step.iteration != 0u) {
      if (index == 0u) {
        return Status::fail(Reason::PipelineInvalid);
      }
      const PipelineBuildStep &previous = build.steps[index - 1u];
      const PipelineWorkspaceRoute previous_route =
          plan.workspace_routes[index - 1u];
      if (previous.program != step.program ||
          previous.logical_step != step.logical_step ||
          previous.iteration_bound != step.iteration_bound ||
          previous.nested != step.nested || previous.route != step.route ||
          previous.iteration == std::numeric_limits<std::uint32_t>::max() ||
          step.iteration != previous.iteration + 1u ||
          previous_route.present() != present) {
        return Status::fail(Reason::PipelineInvalid);
      }
      plan.workspace_routes[index] = previous_route;
      continue;
    }
    if (present) {
      plan.workspace_routes[index].owner = index;
    }
  }
  return Status::success();
}
} // namespace rund::compute::detail

#include "../../state/assembly.hpp"
#include "local.hpp"
#include "accel/local.hpp"

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
plan_pipeline_accel_preparation(const PipelineBuildState &build,
                                PipelineMemoryPlan &plan) {
  plan.accel_preparation = {};
  if (build.device->backend == Backend::Cpu) {
    return Status::success();
  }
  const DeviceOps *const ops = build.device->ops;
  if (ops == nullptr || ops->plan_pipeline_preparation == nullptr ||
      plan.job_owners.size() != build.steps.size() ||
      plan.workspace_routes.size() != build.steps.size() ||
      plan.views.size() != build.steps.size() ||
      plan.window_states.size() != build.steps.size()) {
    return Status::fail(Reason::DeviceInvalid);
  }

  try {
    PipelineAccelPreparationDraft draft{.ops = ops};
    const Status occurrences =
        collect_pipeline_accel_occurrences(build, plan, draft);
    if (!occurrences) {
      return occurrences;
    }
    const Status routes = collect_pipeline_accel_routes(build, plan, draft);
    if (!routes) {
      return routes;
    }
    return finalize_pipeline_accel_preparation(build, plan, draft);
  } catch (const std::bad_alloc &) {
    return Status::fail(Reason::PipelineCapacity);
  }
}
} // namespace rund::compute::detail

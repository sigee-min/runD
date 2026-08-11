#include "local.hpp"

#include "src/compute/pipeline/residency/planner.hpp"

#include <array>

namespace rund_node_test_pipeline_residency {

int CheckGraphPlanner() {
  using namespace rund::compute::detail::residency;
  const PageKey a{.resource = 1u, .page = 0u};
  const PageKey b{.resource = 1u, .page = 1u};
  const PageKey c{.resource = 2u, .page = 0u};
  GraphPlanInput input{
      .page_bytes = 4096u,
      .frame_capacity = 2u,
      .epochs =
          {
              DemandEpoch{.node = 0u,
                          .uses = {{.key = b, .access = Access::Write},
                                   {.key = a, .access = Access::Read}}},
              DemandEpoch{.node = 1u,
                          .uses = {{.key = c, .access = Access::Read},
                                   {.key = a, .access = Access::Read}}},
              DemandEpoch{.node = 2u,
                          .uses = {{.key = b, .access = Access::Read},
                                   {.key = c, .access = Access::Read}}},
          },
  };
  auto planned = PlanResidency(input);
  if (!planned || planned.plan.streamed() ||
      planned.plan.frame_capacity() != 2u ||
      planned.plan.epochs().size() != 3u || planned.plan.uses().size() != 6u ||
      !planned.plan.identity()) {
    return 1;
  }
  const auto &transitions = planned.plan.transitions();
  bool saw_dirty_writeback = false;
  for (std::size_t index = 0u; index < transitions.size(); ++index) {
    if (transitions[index].key == b &&
        transitions[index].kind == TransitionKind::Writeback) {
      if (index + 1u >= transitions.size() ||
          transitions[index + 1u].key != b ||
          transitions[index + 1u].frame != transitions[index].frame ||
          transitions[index + 1u].kind != TransitionKind::Unmap) {
        return 2;
      }
      saw_dirty_writeback = true;
    }
  }
  if (!saw_dirty_writeback) {
    return 3;
  }
  auto same = PlanResidency(input);
  if (!same || same.plan.identity() != planned.plan.identity() ||
      same.plan.transitions() != planned.plan.transitions()) {
    return 4;
  }
  input.frame_capacity = 1u;
  return PlanResidency(input).failure == Failure::Infeasible ? 0 : 5;
}

} // namespace rund_node_test_pipeline_residency

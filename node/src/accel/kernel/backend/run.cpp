#include "run/internal.hpp"

#include "../../backend/token.hpp"

#include <utility>

namespace rund::node::accel::detail {

BoundRun BuildBoundRun(const rund::AccelContext &context,
                       const KernelExecution &execution, const RunBinds &binds,
                       const BoundResets &resets,
                       const PlannedStepStorage &planned,
                       const ScheduledStepOrder &order,
                       const std::uint64_t original_dispatch_count,
                       const std::uint64_t final_dispatch_count) {
  BoundRun result{};
  if (!order.ok || order.size() != execution.steps.size() || !planned.valid() ||
      planned.size() != execution.steps.size() ||
      !execution.context_admission.check.ok ||
      execution.context_admission.pick == nullptr ||
      execution.context_admission.pick->ops == nullptr ||
      !execution.context_admission.pick->raw.check.ok ||
      execution.context_admission.pick->raw.api != context.api ||
      execution.context_admission.pick->ops->api != context.api) {
    return result;
  }
  result.storage.reserve(order.size());
  std::size_t reset_cursor = 0u;
  for (std::size_t position = 0u; position < order.size(); ++position) {
    const std::size_t index = order.at(position);
    if (index >= execution.steps.size()) {
      return result;
    }
    const KernelExecutionStep &step = execution.steps[index];
    const PlannedStep *const step_plan = planned.get(index);
    if (step_plan == nullptr) {
      return result;
    }
    if (reset_cursor < resets.size() &&
        resets[reset_cursor].step.index < index) {
      return result;
    }
    const std::size_t reset_begin = reset_cursor;
    while (reset_cursor < resets.size() &&
           resets[reset_cursor].step.index == index) {
      ++reset_cursor;
    }
    BoundStep bound{.index = index,
                    .step = &step,
                    .planned = step_plan,
                    .source_binds = &binds,
                    .resets =
                        ResetSpan{
                            .begin = reset_begin,
                            .count = reset_cursor - reset_begin,
                        },
                    .barrier_before = order.barrier_before(position)};
    if (!backend_run_detail::bind_primitive(step, binds, bound.bindings) ||
        !backend_run_detail::bind_control(step, binds, bound.control)) {
      return result;
    }
    result.storage.push_back(std::move(bound), order.size());
    BoundStep *const stored = result.storage.get(position);
    if (stored == nullptr || !backend_run_detail::bind_map(context, *stored)) {
      return result;
    }
  }
  if (!result.storage.valid() || reset_cursor != resets.size()) {
    return result;
  }
  result.bind(execution, original_dispatch_count, final_dispatch_count);
  if (result.run.pick == nullptr || result.run.ops == nullptr) {
    return result;
  }
  result.ok = true;
  result.reason = "ok";
  return result;
}

} // namespace rund::node::accel::detail

#include "reduce.hpp"

#include "../../../../reduce/shape.hpp"
#include "../../../bindings/reduce.hpp"

#include <kernel/program/compute/reduce/plan.hpp>

namespace rund::node::accel::detail::device_vsm_reduce_projection {
namespace {

[[nodiscard]] bool same_plan(const rund::kernel::ReducePlan &left,
                             const rund::kernel::ReducePlan &right) noexcept {
  return left.ok && right.ok && left.op == right.op &&
         left.element == right.element &&
         left.element_count == right.element_count &&
         left.element_bytes == right.element_bytes &&
         left.block_size == right.block_size &&
         left.items_per_thread == right.items_per_thread &&
         left.first_pass_group_count == right.first_pass_group_count &&
         left.pass_count == right.pass_count &&
         left.partial_element_count == right.partial_element_count &&
         left.partial_element_bytes == right.partial_element_bytes &&
         left.partial_bytes == right.partial_bytes &&
         left.status_bytes == right.status_bytes &&
         left.temp_bytes == right.temp_bytes &&
         left.count_source == right.count_source;
}

[[nodiscard]] bool exact_step(const prepared::RunState *const run,
                              Authority &authority, const bool graph,
                              const char *&reason) noexcept {
  authority = {};
  if (run == nullptr || !run->bound.ok) {
    reason = "device_vsm_reduce_run_invalid";
    return false;
  }
  if (run->bound.run.step_count != 1u || run->bound.run.steps == nullptr) {
    reason = "device_vsm_reduce_step_count_invalid";
    return false;
  }
  if (!graph && run->bound.run.resets != nullptr &&
      !run->bound.run.resets->empty()) {
    reason = "device_vsm_reduce_reset_invalid";
    return false;
  }
  const BoundStep &step = run->bound.run.steps[0u];
  const auto *const active = OperationFor<operation::Reduce>(step);
  const auto *const bindings =
      BindingsFor<ReduceBinds>(step, rund::kernel::NodeKind::Reduce);
  const rund::kernel::ComputeApi api =
      run->execution.context_admission.api == rund::AccelApi::Metal
          ? rund::kernel::ComputeApi::Metal
      : run->execution.context_admission.api == rund::AccelApi::Vulkan
          ? rund::kernel::ComputeApi::Vulkan
          : rund::kernel::ComputeApi::Cpu;
  if (active == nullptr || bindings == nullptr) {
    reason = "device_vsm_reduce_operation_invalid";
    return false;
  }
  if ((!graph && step.control.active()) ||
      api == rund::kernel::ComputeApi::Cpu) {
    reason = "device_vsm_reduce_control_invalid";
    return false;
  }
  if (!ReduceShapeOk(active->desc, active->plan, *bindings)) {
    reason = "device_vsm_reduce_shape_invalid";
    return false;
  }
  if (active->plan.op != rund::kernel::ReduceOp::Sum &&
      active->plan.op != rund::kernel::ReduceOp::CountNonzero &&
      active->plan.op != rund::kernel::ReduceOp::Min &&
      active->plan.op != rund::kernel::ReduceOp::Max) {
    reason = "device_vsm_reduce_operation_unsupported";
    return false;
  }
  if (active->plan.element != rund::kernel::ReduceElement::U32 &&
      active->plan.element != rund::kernel::ReduceElement::U64) {
    reason = "device_vsm_reduce_element_unsupported";
    return false;
  }
  const rund::kernel::ComputeCountSource expected_count =
      graph ? rund::kernel::ComputeCountSource::BufferU64
            : rund::kernel::ComputeCountSource::Descriptor;
  if (active->plan.count_source != expected_count) {
    reason = "device_vsm_reduce_count_source_invalid";
    return false;
  }
  authority = Authority{.semantic = active->plan, .api = api};
  return true;
}

[[nodiscard]] bool exact_pipeline(const prepared::PipelineState &pipeline,
                                  Authority &authority, const bool graph,
                                  const char *&reason) noexcept {
  authority = {};
  reason = graph ? "device_vsm_graph_reduce_pipeline_invalid"
                 : "device_vsm_reduce_pipeline_invalid";
  if (pipeline.state_count == 0u || pipeline.states == nullptr ||
      pipeline.size == 0u ||
      !exact_step(pipeline.states[0u].get(), authority, graph, reason)) {
    return false;
  }
  for (std::size_t index = 1u; index < pipeline.state_count; ++index) {
    Authority candidate{};
    if (!exact_step(pipeline.states[index].get(), candidate, graph, reason) ||
        candidate.api != authority.api ||
        !same_plan(candidate.semantic, authority.semantic)) {
      if (candidate.semantic.ok) {
        reason = graph ? "device_vsm_graph_reduce_state_mismatch"
                       : "device_vsm_reduce_state_mismatch";
      }
      return false;
    }
  }
  reason = "ok";
  return true;
}

} // namespace

bool exact(const prepared::PipelineState &pipeline, Authority &authority,
           const char *&reason) noexcept {
  return exact_pipeline(pipeline, authority, false, reason);
}

bool exact_graph(const prepared::PipelineState &pipeline, Authority &authority,
                 const char *&reason) noexcept {
  return exact_pipeline(pipeline, authority, true, reason);
}

bool same(const prepared::PipelineState &pipeline,
          const Authority &authority) noexcept {
  Authority candidate{};
  const char *reason = nullptr;
  return exact(pipeline, candidate, reason) && candidate.api == authority.api &&
         same_plan(candidate.semantic, authority.semantic);
}

} // namespace rund::node::accel::detail::device_vsm_reduce_projection

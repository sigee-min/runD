#pragma once

#include "artifact.hpp"

namespace rund::node::accel::detail {

#if defined(__APPLE__) && defined(RUND_NODE_HAVE_METAL_SDK)

[[nodiscard]] inline bool
MetalSpatialWindowSameRangeShape(const RangeShape &left,
                                 const RangeShape &right) noexcept {
  return left.traits().operation() == right.traits().operation() &&
         left.traits().domain() == right.traits().domain() &&
         left.traits().arithmetic_law() == right.traits().arithmetic_law() &&
         left.boundary() == right.boundary() && left.count() == right.count() &&
         left.input_count() == right.input_count() &&
         left.output_count() == right.output_count() &&
         left.window_size() == right.window_size() &&
         left.stride() == right.stride() && left.padding() == right.padding() &&
         left.element_bytes() == right.element_bytes();
}

[[nodiscard]] inline bool
MetalSpatialWindowSameRangePlan(const RangePlan &left,
                                const RangePlan &right) noexcept {
  if (left.disposition() != right.disposition() ||
      !SameReason(left.reason(), right.reason())) {
    return false;
  }
  if (!left.ok() || !right.ok()) {
    return true;
  }
  if (!MetalSpatialWindowSameRangeShape(left.shape(), right.shape()) ||
      left.candidate() != right.candidate() ||
      left.source_variant() != right.source_variant() ||
      left.cost() != right.cost() ||
      left.stage_count() != right.stage_count() ||
      left.temporary_count() != right.temporary_count() ||
      left.legal_candidate_count() != right.legal_candidate_count() ||
      left.pareto_candidate_count() != right.pareto_candidate_count() ||
      left.source_identity() != right.source_identity() ||
      left.execution_identity() != right.execution_identity()) {
    return false;
  }
  for (std::size_t index = 0u; index < left.stage_count(); ++index) {
    if (left.stage(index) != right.stage(index)) {
      return false;
    }
  }
  for (std::size_t index = 0u; index < left.temporary_count(); ++index) {
    if (left.temporary(index) != right.temporary(index)) {
      return false;
    }
  }
  return true;
}

[[nodiscard]] inline bool
MetalSpatialWindowPlannedStepValid(const BoundStep &bound,
                                   const KernelExecutionStep &step) noexcept {
  if (bound.step != &step || bound.planned == nullptr ||
      !bound.planned->plan.ok || !BoundStepMatches(bound, step.kind())) {
    return false;
  }
  if (step.kind() == rund::kernel::NodeKind::Map) {
    return bound.planned->artifact == &step.artifact;
  }
  if (step.kind() != rund::kernel::NodeKind::Window) {
    return false;
  }
  const operation::Window &window = step.operation.get<operation::Window>();
  return bound.planned->artifact == nullptr && window.plan.ok &&
         window.range.ok() &&
         bound.planned->plan.dispatch_count == window.range.stage_count();
}

[[nodiscard]] inline bool MetalSpatialWindowSameWindowDescriptor(
    const rund::kernel::WindowDesc &left,
    const rund::kernel::WindowDesc &right) noexcept {
  return left.op == right.op && left.element == right.element &&
         left.boundary == right.boundary && left.domain == right.domain &&
         left.fixed_format == right.fixed_format &&
         left.count_source == right.count_source &&
         left.input_count == right.input_count &&
         left.output_count == right.output_count &&
         left.window_size == right.window_size && left.stride == right.stride &&
         left.pad_left == right.pad_left;
}

[[nodiscard]] inline bool MetalSpatialWindowSameWindowPlan(
    const rund::kernel::WindowPlan &left,
    const rund::kernel::WindowPlan &right) noexcept {
  return left.op == right.op && left.element == right.element &&
         left.boundary == right.boundary && left.domain == right.domain &&
         left.fixed_format == right.fixed_format &&
         left.count_source == right.count_source &&
         left.input_count == right.input_count &&
         left.output_count == right.output_count &&
         left.window_size == right.window_size && left.stride == right.stride &&
         left.pad_left == right.pad_left &&
         left.element_bytes == right.element_bytes &&
         left.input_bytes == right.input_bytes &&
         left.output_bytes == right.output_bytes &&
         left.temp_bytes == right.temp_bytes &&
         left.pass_count == right.pass_count && left.ok == right.ok &&
         SameReason(left.reason, right.reason);
}

[[nodiscard]] inline bool
MetalSpatialWindowSameOperation(const Operation &left,
                                const Operation &right) noexcept {
  if (left.kind() != right.kind()) {
    return false;
  }
  if (left.kind() == rund::kernel::NodeKind::Map) {
    return true;
  }
  if (left.kind() != rund::kernel::NodeKind::Window) {
    return false;
  }
  const operation::Window &left_window = left.get<operation::Window>();
  const operation::Window &right_window = right.get<operation::Window>();
  return MetalSpatialWindowSameWindowDescriptor(left_window.desc,
                                                right_window.desc) &&
         MetalSpatialWindowSameWindowPlan(left_window.plan,
                                          right_window.plan) &&
         MetalSpatialWindowSameRangePlan(left_window.range, right_window.range);
}

#endif

} // namespace rund::node::accel::detail

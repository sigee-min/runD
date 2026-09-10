#include "local.hpp"

#include "../../../../recurrence/match.hpp"

namespace rund::node::accel::detail::device_vsm_window_projection::
    window_detail {
namespace {

[[nodiscard]] bool same_plan(const rund::kernel::WindowPlan &left,
                             const rund::kernel::WindowPlan &right) {
  return left.ok && right.ok && left.op == right.op &&
         left.element == right.element && left.boundary == right.boundary &&
         left.domain == right.domain &&
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
         left.pass_count == right.pass_count;
}

} // namespace

bool same_authority(const WindowAuthority &first,
                    const WindowAuthority &candidate) {
  if (first.step == nullptr || first.active == nullptr ||
      candidate.step == nullptr || candidate.active == nullptr) {
    return false;
  }
  const RangePlan *const left = RangePlanFor(first.step->step->operation);
  const RangePlan *const right = RangePlanFor(candidate.step->step->operation);
  const auto same_map = [](const WindowMapAuthority &a,
                           const WindowMapAuthority &b) {
    if (!a.map.active() || !b.map.active()) {
      return !a.map.active() && !b.map.active();
    }
    return a.step != nullptr && b.step != nullptr && a.map == b.map &&
           SamePlan(a.step->planned->plan, b.step->planned->plan) &&
           SameArtifact(a.step->step->artifact, b.step->step->artifact) &&
           SameWindows(*a.step, *b.step) &&
           SameMapBinding(a.bindings, b.bindings) &&
           same_parameters(a.bindings, b.bindings);
  };
  return left != nullptr && right != nullptr &&
         same_plan(first.active->plan, candidate.active->plan) &&
         first.fusion == candidate.fusion &&
         same_map(first.before, candidate.before) &&
         same_map(first.before_second, candidate.before_second) &&
         same_map(first.before_third, candidate.before_third) &&
         same_map(first.after, candidate.after) &&
         same_map(first.after_second, candidate.after_second) &&
         same_map(first.after_third, candidate.after_third) &&
         left->source_identity() == right->source_identity() &&
         left->execution_identity() == right->execution_identity();
}

} // namespace
  // rund::node::accel::detail::device_vsm_window_projection::window_detail

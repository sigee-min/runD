#include "local.hpp"

#include "../../validation.hpp"

#include "../../../../../range_aggregate/execution/projection.hpp"
#include "../../../../../window/shape.hpp"
#include "../../../../recurrence/match.hpp"

#include <optional>

namespace rund::node::accel::detail::device_vsm_window_projection::
    window_detail {

bool exact_step(const prepared::RunState *const run, WindowAuthority &authority,
                const char *&reason) {
  authority = {};
  reason = "device_vsm_window_run_invalid";
  if (run == nullptr || !run->bound.ok || run->bound.run.step_count == 0u ||
      run->bound.run.step_count > 7u || run->bound.run.steps == nullptr) {
    return false;
  }
  std::size_t cursor = 0u;
  DeviceVsmWindowFusion fusion{
      .stage_count = static_cast<std::uint32_t>(run->bound.run.step_count)};
  WindowMapAuthority before{};
  WindowMapAuthority before_second{};
  WindowMapAuthority before_third{};
  if (!project_prefix_map(*run, cursor, before, fusion.before, reason) ||
      !project_prefix_map(*run, cursor, before_second, fusion.before_second,
                          reason) ||
      !project_prefix_map(*run, cursor, before_third, fusion.before_third,
                          reason)) {
    return false;
  }
  if (cursor >= run->bound.run.step_count) {
    return false;
  }
  const BoundStep &step = run->bound.run.steps[cursor++];
  const auto *const active = OperationFor<operation::Window>(step);
  const auto *const bindings =
      BindingsFor<RangeBinds>(step, rund::kernel::NodeKind::Window);
  const RangePlan *const range =
      active == nullptr ? nullptr : RangePlanFor(step.step->operation);
  const std::optional<RangeExec> execution =
      range == nullptr ? std::nullopt : RangeExec::from(*range);
  const bool has_resets =
      run->bound.run.resets != nullptr && !run->bound.run.resets->empty();
  if (active == nullptr || bindings == nullptr || range == nullptr ||
      !execution.has_value() || step.planned == nullptr) {
    reason = "device_vsm_window_authority_invalid";
    return false;
  }
  if (step.control.active()) {
    reason = "device_vsm_window_control_invalid";
    return false;
  }
  // A reset is an independently scheduled mutation. It cannot be erased by
  // the one-dispatch DeviceVsm proof merely because the generated Window
  // source could express the same boundary value.
  if (has_resets) {
    reason = "device_vsm_window_reset_invalid";
    return false;
  }
  if (!WindowShapeOk(active->desc, active->plan, *bindings)) {
    reason = "device_vsm_window_shape_invalid";
    return false;
  }
  if (!WindowRangePlanMatches(active->plan, *range)) {
    reason = "device_vsm_window_range_invalid";
    return false;
  }
  if (!DeviceVsmWindowRangeSupported(*execution)) {
    reason = "device_vsm_window_candidate_invalid";
    return false;
  }
  WindowMapAuthority after{};
  WindowMapAuthority after_second{};
  WindowMapAuthority after_third{};
  if (!project_suffix_map(*run, cursor, after, fusion.after, reason) ||
      !project_suffix_map(*run, cursor, after_second, fusion.after_second,
                          reason) ||
      !project_suffix_map(*run, cursor, after_third, fusion.after_third,
                          reason)) {
    return false;
  }
  if (cursor != run->bound.run.step_count ||
      fusion.stage_count !=
          1u + static_cast<std::uint32_t>(fusion.before.active()) +
              static_cast<std::uint32_t>(fusion.before_second.active()) +
              static_cast<std::uint32_t>(fusion.before_third.active()) +
              static_cast<std::uint32_t>(fusion.after.active()) +
              static_cast<std::uint32_t>(fusion.after_second.active()) +
              static_cast<std::uint32_t>(fusion.after_third.active())) {
    reason = "device_vsm_window_fusion_invalid";
    return false;
  }
  authority = WindowAuthority{.step = &step,
                              .active = active,
                              .bindings = bindings,
                              .before = before,
                              .before_second = before_second,
                              .before_third = before_third,
                              .after = after,
                              .after_second = after_second,
                              .after_third = after_third,
                              .fusion = fusion};
  reason = "ok";
  return true;
}

} // namespace
  // rund::node::accel::detail::device_vsm_window_projection::window_detail

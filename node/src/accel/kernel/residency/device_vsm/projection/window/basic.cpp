#include "local.hpp"

#include <cstring>

namespace rund::node::accel::detail::device_vsm_window_projection {

const BoundStep *map_step(const prepared::RunState *const run,
                          const bool graph) {
  if (run == nullptr || !run->bound.ok || run->bound.run.step_count != 1u ||
      run->bound.run.steps == nullptr ||
      (!graph && run->bound.run.resets != nullptr &&
       !run->bound.run.resets->empty())) {
    return nullptr;
  }
  const BoundStep &step = run->bound.run.steps[0u];
  return BoundStepMatches(step, rund::kernel::NodeKind::Map) &&
                 (graph || !step.control.active()) && step.step != nullptr &&
                 step.step->map_semantic.recurrence_total &&
                 step.planned != nullptr && step.planned->artifact != nullptr &&
                 step.planned->artifact == &step.step->artifact &&
                 step.step->artifact.metadata.read_routes.empty() &&
                 step.map_windows.ok && step.map_windows.size() != 0u
             ? &step
             : nullptr;
}

bool same_parameters(const rund::kernel::BindingSet &left,
                     const rund::kernel::BindingSet &right) {
  return left.param_bytes == right.param_bytes &&
         left.param_data_bytes == right.param_data_bytes &&
         left.param_data_bytes == left.param_bytes &&
         (left.param_bytes == 0u ||
          (left.param_data != nullptr && right.param_data != nullptr &&
           std::memcmp(left.param_data, right.param_data,
                       static_cast<std::size_t>(left.param_bytes)) == 0));
}

DeviceVsmWindowMapSources map_sources(const WindowAuthority &window) noexcept {
  const auto map_source = [](const WindowMapAuthority &map) {
    return map.map.kind == DeviceVsmWindowMapKind::CanonicalTotalU32 &&
                   map.step != nullptr
               ? DeviceVsmWindowMapSource{
                     .artifact = &map.step->step->artifact,
                     .input = &map.step->step->cpu_input,
                 }
               : DeviceVsmWindowMapSource{};
  };
  return DeviceVsmWindowMapSources{
      .before = map_source(window.before),
      .before_second = map_source(window.before_second),
      .before_third = map_source(window.before_third),
      .after = map_source(window.after),
      .after_second = map_source(window.after_second),
      .after_third = map_source(window.after_third),
  };
}

} // namespace rund::node::accel::detail::device_vsm_window_projection

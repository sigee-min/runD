#include "internal.hpp"

#include "../../step/map/local.hpp"

#include <accel/api.hpp>

namespace rund::node::accel::detail::backend_run_detail {

bool bind_map(const rund::AccelContext &context, BoundStep &bound) {
  if (bound.step == nullptr || bound.planned == nullptr ||
      bound.step->kind() != rund::kernel::NodeKind::Map) {
    return true;
  }
  const StepBinds *const bindings = BindingsFor<StepBinds>(bound);
  if (bindings == nullptr) {
    return false;
  }
  const rund::kernel::BindingSet map_binding =
      BindMapStep(*bound.step, *bound.planned, *bindings);
  const rund::kernel::BindingValidation validation =
      ValidateMapStepBindings(*bound.step, *bound.planned, map_binding);
  if (!validation.ok) {
    return false;
  }
  const bool resident_identity = map_binding.has_resident_output() &&
                                 map_binding.sequence_tiles == nullptr &&
                                 map_binding.sequence_tile_count == 0u;
  bound.map_windows =
      ResidentDispatchWindows(bound.planned->plan, resident_identity,
                              context.api != rund::AccelApi::Fake &&
                                  context.api != rund::AccelApi::Cpu);
  return bound.map_windows.ok;
}

} // namespace rund::node::accel::detail::backend_run_detail

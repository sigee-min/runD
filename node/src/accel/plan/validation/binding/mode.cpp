#include "../local.hpp"

namespace rund::node::accel::detail {

bool BindingInputCountMatches(
    const rund::kernel::ComputePlan& plan,
    const rund::kernel::BindingSet& bindings,
    const PlanBindingInputMode input_mode) noexcept {
  if (input_mode == PlanBindingInputMode::ResidentInputs ||
      input_mode == PlanBindingInputMode::MapResidentViews) {
    return bindings.resident_inputs.count == plan.input_buffer_count;
  }
  return bindings.has_resident_output()
             ? bindings.resident_inputs.count == plan.input_buffer_count
             : bindings.input_buffer_count == plan.input_buffer_count;
}

}  // namespace rund::node::accel::detail

#include "../internal.hpp"
#include "model.hpp"

namespace rund::compute::detail {

bool project_virtual_graph_run(VirtualPipelineState &state,
                               const std::uint64_t active_count,
                               VirtualRunProjection &projection) noexcept {
  GraphProjectionInputs inputs{};
  if (!validate_virtual_graph_projection(state, inputs)) {
    return false;
  }

  VirtualActiveProjection active{};
  if (!project_virtual_graph_active(
          *inputs.capacity_plan, active_count, inputs.input->count,
          state.geometry.input_payload_elements, inputs.input->element_bytes,
          state.output->element_bytes, active)) {
    return false;
  }

  GraphProjectionBanks banks{};
  return bind_virtual_graph_banks(inputs, banks) &&
         materialize_virtual_graph_projection(state, inputs, banks, active,
                                              projection);
}

} // namespace rund::compute::detail

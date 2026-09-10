#include "local.hpp"

#include "../../pipeline/local.hpp"

namespace rund::compute::detail::virtual_observe {

bool valid_input_authority(const VirtualPipelineState &state,
                           const std::size_t minimum,
                           const std::size_t maximum) noexcept {
  if (state.input_count < minimum || state.input_count > maximum ||
      state.input_count > state.inputs.size()) {
    return false;
  }
  for (std::size_t index = 0u; index < state.inputs.size(); ++index) {
    if (index >= state.input_count) {
      if (state.inputs[index] != nullptr) {
        return false;
      }
      continue;
    }
    if (state.inputs[index] == nullptr ||
        state.inputs[index]->backing == nullptr ||
        state.inputs[index] == state.output ||
        (state.output != nullptr &&
         state.inputs[index]->backing == state.output->backing)) {
      return false;
    }
    for (std::size_t prior = 0u; prior < index; ++prior) {
      if (state.inputs[index] == state.inputs[prior] ||
          state.inputs[index]->backing == state.inputs[prior]->backing) {
        return false;
      }
    }
  }
  return true;
}

bool valid_poolless_device_vsm(const VirtualPipelineState &state) noexcept {
  const PipelineState *const first = state.pipeline.get();
  const PipelineState *const second = state.alternate_pipeline.get();
  const bool pointwise = state.geometry.route == VirtualRoute::MultiPointwise;
  const bool scan = state.geometry.route == VirtualRoute::Scan;
  constexpr std::size_t expected_steps = 1u;
  const ProgramState *const first_program =
      first == nullptr || first->steps.size() != expected_steps
          ? nullptr
          : first->steps.front().program.get();
  const ProgramState *const second_program =
      second == nullptr || second->steps.size() != expected_steps
          ? nullptr
          : second->steps.front().program.get();
  return (pointwise || scan) && state.geometry.device_vsm_required &&
         valid_input_authority(state, 2u,
                               VirtualPipelineState::InputCapacity) &&
         state.output != nullptr && first != nullptr && second != nullptr &&
         first != second && valid_pipeline(state.pipeline) &&
         valid_pipeline(state.alternate_pipeline) && first->device != nullptr &&
         first->device == second->device &&
         first->device->backend != Backend::Cpu &&
         first->residency == nullptr && second->residency == nullptr &&
         first->residency_pool == nullptr &&
         second->residency_pool == nullptr && !first->transactional &&
         !second->transactional && first_program != nullptr &&
         second_program != nullptr &&
         first->logical_step_count == expected_steps &&
         second->logical_step_count == expected_steps &&
         first_program->graph_info.fingerprint ==
             second_program->graph_info.fingerprint &&
         first_program->input_types.size() == state.input_count &&
         first_program->output_types.size() == 1u &&
         state.graph_pipelines.empty() &&
         state.graph_input_resource_count == 0u &&
         state.device_vsm_semantic_pipeline == nullptr;
}

} // namespace rund::compute::detail::virtual_observe

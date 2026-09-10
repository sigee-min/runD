#include "validation/local.hpp"

namespace rund::kernel {

using binding_validation_detail::Reject;

BindingValidation
ValidateRuntimeBindings(const BindingSet &bindings,
                        const BindingObligations &obligations) noexcept {
  if (!bindings.ok) {
    return Reject(bindings.reason);
  }
  if (obligations.tile_count == 0u ||
      bindings.tile_count != obligations.tile_count) {
    return Reject("compute_binding_tile_count_mismatch");
  }
  const bool has_resident_inputs = !bindings.resident_inputs.empty();
  const bool has_resident_output = !bindings.resident_outputs.empty();

  if (bindings.resident_inputs.count == 0u && has_resident_inputs) {
    return Reject("compute_binding_input_null");
  }
  if (bindings.resident_outputs.count == 0u && has_resident_output) {
    return Reject("compute_binding_output_missing");
  }
  if (!has_resident_inputs &&
      bindings.input_buffer_count != obligations.input_buffer_count) {
    return Reject("compute_binding_input_count_mismatch");
  }
  if (has_resident_inputs &&
      bindings.resident_inputs.count != obligations.input_buffer_count) {
    return Reject("compute_binding_input_count_mismatch");
  }
  if (bindings.input_bytes_per_tile != obligations.input_bytes_per_tile) {
    return Reject("compute_binding_input_bytes_mismatch");
  }
  if (obligations.validate_staged_output &&
      bindings.output_bytes_per_tile != obligations.output_bytes_per_tile) {
    return Reject("compute_binding_output_bytes_mismatch");
  }
  if (bindings.param_bytes != obligations.param_bytes) {
    return Reject("compute_binding_param_size_mismatch");
  }

  const BindingValidation inputs =
      binding_validation_detail::ValidateInputs(bindings, obligations);
  if (!inputs) {
    return inputs;
  }
  if (obligations.param_bytes != bindings.param_data_bytes) {
    return Reject("compute_binding_param_size_mismatch");
  }
  if (obligations.param_bytes != 0u && bindings.param_data == nullptr) {
    return Reject("compute_binding_param_null");
  }
  return binding_validation_detail::ValidateOutputs(bindings, obligations);
}

} // namespace rund::kernel

#include "model.hpp"

#include "../../buffer/local.hpp"
#include "../../pipeline/claim.hpp"

#include <algorithm>

namespace rund::compute::detail {

Status validate_host_inputs(const ProgramState &program,
                            const std::span<const HostView> inputs) noexcept {
  if (!valid_input_shape(program)) {
    return Status::fail(Reason::ProgramInputShapeInvalid);
  }
  if (inputs.size() != program.input_types.size()) {
    return Status::fail(Reason::BindingCountMismatch);
  }
  for (std::size_t index = 0u; index < inputs.size(); ++index) {
    if (inputs[index].type != program.input_types[index] ||
        inputs[index].count != program.input_sizes[index] ||
        (inputs[index].data == nullptr && inputs[index].count != 0u)) {
      return Status::fail(Reason::ShapeMismatch);
    }
    const Status bounded =
        validate_host_bounded_input(program, index, inputs[index]);
    if (!bounded) {
      return bounded;
    }
  }
  return Status::success();
}

Status validate_bound_buffers(
    const std::shared_ptr<ProgramState> &program,
    const std::span<const std::shared_ptr<BufferState>> inputs,
    const std::span<const std::shared_ptr<BufferState>> outputs,
    const bool check_poison) noexcept {
  if (program == nullptr || program->device == nullptr) {
    return Status::fail(Reason::ProgramInvalid);
  }
  if (!valid_input_shape(*program)) {
    return Status::fail(Reason::ProgramInputShapeInvalid);
  }
  if (inputs.size() != program->input_types.size() ||
      outputs.size() != program->output_types.size() ||
      outputs.size() != program->output_sizes.size() || outputs.empty() ||
      outputs.size() > MaxOutputs) {
    return Status::fail(Reason::BindingCountMismatch);
  }
  for (std::size_t index = 0u; index < inputs.size(); ++index) {
    const auto &input = inputs[index];
    if (input == nullptr) {
      return Status::fail(Reason::BindingCountMismatch);
    }
    if (check_poison && buffer_poisoned(*input)) {
      return Status::fail(Reason::BufferPoisoned);
    }
    if (input->device != program->device) {
      return Status::fail(Reason::BindingDeviceMismatch);
    }
    if (input->type != program->input_types[index]) {
      return Status::fail(Reason::BindingTypeMismatch);
    }
    if (input->count != program->input_sizes[index]) {
      return Status::fail(Reason::ShapeMismatch);
    }
    if (std::find(outputs.begin(), outputs.end(), input) != outputs.end()) {
      return Status::fail(Reason::BindingAliasUnsupported);
    }
  }
  for (std::size_t index = 0u; index < outputs.size(); ++index) {
    const auto &output = outputs[index];
    if (output == nullptr) {
      return Status::fail(Reason::BindingCountMismatch);
    }
    if (check_poison && buffer_poisoned(*output)) {
      return Status::fail(Reason::BufferPoisoned);
    }
    if (output->device != program->device) {
      return Status::fail(Reason::BindingDeviceMismatch);
    }
    if (output->type != program->output_types[index]) {
      return Status::fail(Reason::BindingTypeMismatch);
    }
    if (output->count != program->output_sizes[index]) {
      return Status::fail(Reason::ShapeMismatch);
    }
  }
  return Status::success();
}

bool same_buffers(
    const std::shared_ptr<JobState> &state,
    const std::span<const std::shared_ptr<BufferState>> inputs,
    const std::span<const std::shared_ptr<BufferState>> outputs) noexcept {
  return state != nullptr && state->inputs.size() == inputs.size() &&
         state->outputs.size() == outputs.size() &&
         std::equal(state->inputs.begin(), state->inputs.end(),
                    inputs.begin()) &&
         std::equal(state->outputs.begin(), state->outputs.end(),
                    outputs.begin());
}

} // namespace rund::compute::detail

#include "internal.hpp"

#include "../../backend.hpp"
#include "../local.hpp"

#include <algorithm>

namespace rund::compute::detail {

Status validate_virtual_pipeline_program(
    const ProgramState &program,
    const std::span<const std::shared_ptr<VirtualBufferState>> inputs,
    const VirtualBufferState &output) noexcept {
  if (program.device == nullptr || inputs.empty() ||
      inputs.size() > VirtualPipelineState::InputCapacity ||
      program.input_types.size() != inputs.size() ||
      program.input_sizes.size() != inputs.size() ||
      program.input_formats.size() != inputs.size() ||
      program.output_types.size() != 1u || program.output_sizes.size() != 1u ||
      program.output_formats.size() != 1u || program.input_sizes[0] == 0u ||
      program.output_types[0] != output.type ||
      program.output_formats[0] != output.format) {
    return Status::fail(Reason::PrimitiveUnsupported);
  }
  const VirtualBufferState *const first = inputs.front().get();
  if (first == nullptr || first->count == 0u) {
    return Status::fail(Reason::PrimitiveUnsupported);
  }
  for (std::size_t index = 0u; index < inputs.size(); ++index) {
    const VirtualBufferState *const input = inputs[index].get();
    if (input == nullptr || program.input_types[index] != input->type ||
        program.input_formats[index] != input->format ||
        input->count != first->count) {
      return Status::fail(Reason::PrimitiveUnsupported);
    }
  }
  const auto geometry = virtual_prepare_detail::classify_geometry(program);
  if (!geometry || ((geometry->route == VirtualRoute::Reduction ||
                     geometry->route == VirtualRoute::GraphReduction)
                        ? output.count != 1u
                        : first->count != output.count)) {
    return Status::fail(Reason::PrimitiveUnsupported);
  }
  if (geometry->device_vsm_required &&
      program.device->backend == Backend::Cpu) {
    return Status::fail(Reason::PrimitiveUnsupported);
  }
  return Status::success();
}

} // namespace rund::compute::detail

namespace rund::compute::detail::virtual_prepare_detail {

Status validate_pipeline_capability(const DeviceState &device) noexcept {
  if (device.backend == Backend::Cpu) {
    return Status::success();
  }
  if (device.ops == nullptr ||
      device.ops->virtual_execution.virtual_pipeline_capability == nullptr) {
    return Status::fail(Reason::BackendUnsupported);
  }
  return device.ops->virtual_execution.virtual_pipeline_capability(device);
}

} // namespace rund::compute::detail::virtual_prepare_detail

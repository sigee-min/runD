#include "internal.hpp"

#include "../../graph/compile/slice/semantic.hpp"
#include "../../type.hpp"

#include <kernel/core/checked.hpp>
#include <kernel/program/compute/scan/model.hpp>

#include <algorithm>
#include <limits>

namespace rund::compute::detail::virtual_multi_detail {
namespace {

static_assert(graph_compile::ServiceFreeMapScanInputCapacity ==
              VirtualPipelineState::InputCapacity);

[[nodiscard]] bool distinct_owners(
    const std::span<const std::shared_ptr<VirtualBufferState>> inputs,
    const std::shared_ptr<VirtualBufferState> &output) noexcept {
  for (std::size_t index = 0u; index < inputs.size(); ++index) {
    if (inputs[index] == nullptr || inputs[index] == output ||
        inputs[index]->backing == nullptr ||
        inputs[index]->backing == output->backing) {
      return false;
    }
    for (std::size_t prior = 0u; prior < index; ++prior) {
      if (inputs[index] == inputs[prior] ||
          inputs[index]->backing == inputs[prior]->backing) {
        return false;
      }
    }
  }
  return true;
}

} // namespace

Status
inspect(const std::shared_ptr<ProgramState> &program,
        const std::span<const std::shared_ptr<VirtualBufferState>> inputs,
        const std::shared_ptr<VirtualBufferState> &output,
        const VirtualGeometry &geometry, Schema &schema) noexcept {
  schema = {};
  if (program == nullptr || program->device == nullptr ||
      program->device->backend == Backend::Cpu || output == nullptr ||
      output->backing == nullptr || inputs.size() < 2u ||
      inputs.size() > VirtualPipelineState::InputCapacity ||
      program->input_types.size() != inputs.size() ||
      program->input_sizes.size() != inputs.size() ||
      program->input_formats.size() != inputs.size() ||
      program->output_types.size() != 1u ||
      program->output_sizes.size() != 1u ||
      program->output_formats.size() != 1u ||
      !distinct_owners(inputs, output)) {
    return Status::fail(Reason::PrimitiveUnsupported);
  }
  const Type type = program->input_types.front();
  const std::uint64_t element_bytes = type_bytes(type);
  const std::uint64_t page_elements = geometry.input_payload_elements;
  const std::uint64_t frame_elements = geometry.input_frame_elements;
  const std::uint64_t capacity_count = inputs.front()->count;
  const bool pointwise =
      geometry.route == VirtualRoute::Pointwise &&
      geometry.input_payload_elements == geometry.output_payload_elements &&
      geometry.input_frame_elements == geometry.input_payload_elements &&
      geometry.output_frame_elements == geometry.output_payload_elements &&
      geometry.input_prefix_elements == 0u &&
      geometry.output_prefix_elements == 0u;
  const bool scan =
      geometry.route == VirtualRoute::Scan && geometry.device_vsm_required &&
      inputs.size() <= graph_compile::ServiceFreeMapScanInputCapacity &&
      type == Type::U64 &&
      (geometry.operation ==
           static_cast<std::uint32_t>(kernel::ScanOp::InclusiveSum) ||
       geometry.operation ==
           static_cast<std::uint32_t>(kernel::ScanOp::ExclusiveSum)) &&
      geometry.input_payload_elements == geometry.output_payload_elements &&
      geometry.input_frame_elements == geometry.output_frame_elements &&
      geometry.input_prefix_elements == geometry.output_prefix_elements &&
      geometry.input_prefix_elements + geometry.input_payload_elements ==
          geometry.input_frame_elements;
  if ((!pointwise && !scan) || (type != Type::U32 && type != Type::U64) ||
      element_bytes == 0u || page_elements == 0u || frame_elements == 0u ||
      capacity_count == 0u || program->output_types.front() != type ||
      program->output_formats.front() != FixedFormat{} ||
      program->output_sizes.front() != geometry.output_frame_elements ||
      output->type != type || output->format != FixedFormat{} ||
      output->count != capacity_count ||
      output->element_bytes != element_bytes ||
      output->backing->size_bytes() != output->bytes) {
    return Status::fail(Reason::PrimitiveUnsupported);
  }
  for (std::size_t index = 0u; index < inputs.size(); ++index) {
    const VirtualBufferState &input = *inputs[index];
    if (program->input_types[index] != type ||
        program->input_formats[index] != FixedFormat{} ||
        program->input_sizes[index] != frame_elements || input.type != type ||
        input.format != FixedFormat{} || input.count != capacity_count ||
        input.element_bytes != element_bytes ||
        input.backing->size_bytes() != input.bytes) {
      return Status::fail(Reason::PrimitiveUnsupported);
    }
  }
  const std::uint64_t page_count =
      capacity_count / page_elements +
      static_cast<std::uint64_t>(capacity_count % page_elements != 0u);
  std::uint64_t page_bytes = 0u;
  std::uint64_t logical_bytes = output->bytes;
  if (page_count < 2u ||
      page_count > std::numeric_limits<std::uint32_t>::max() ||
      !kernel::checked::mul(frame_elements, element_bytes, page_bytes)) {
    return Status::fail(Reason::PipelineCapacity);
  }
  for (const std::shared_ptr<VirtualBufferState> &input : inputs) {
    if (!kernel::checked::add(logical_bytes, input->bytes, logical_bytes)) {
      return Status::fail(Reason::PipelineCapacity);
    }
  }
  auto semantic =
      scan ? graph_compile::compile_service_free_map_scan(program)
           : graph_compile::compile_service_free_map_program(program);
  if (!semantic || semantic.value() == nullptr ||
      semantic.value()->input_types.size() != inputs.size() ||
      semantic.value()->output_types.size() != 1u) {
    return Status::fail(semantic ? Reason::ProgramInvalid : semantic.reason());
  }
  VirtualGeometry normalized = geometry;
  normalized.route = scan ? VirtualRoute::Scan : VirtualRoute::MultiPointwise;
  normalized.device_vsm_required = true;
  normalized.materialization_hi = program->graph_info.fingerprint.hi;
  normalized.materialization_lo = program->graph_info.fingerprint.lo;
  schema = Schema{
      .semantic = std::move(semantic).value(),
      .geometry = normalized,
      .page_count = page_count,
      .page_bytes = page_bytes,
      .logical_bytes = logical_bytes,
  };
  return Status::success();
}

} // namespace rund::compute::detail::virtual_multi_detail

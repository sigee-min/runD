#include "../projection.hpp"

namespace rund::compute::detail {

bool bind_virtual_run_transfer(VirtualPipelineState &state,
                               VirtualRunProjection &projection) noexcept {
  return state.pipeline != nullptr
             ? bind_virtual_run_transfer(*state.pipeline, projection)
             : false;
}

bool bind_virtual_run_transfer(PipelineState &pipeline,
                               VirtualRunProjection &projection) noexcept {
  PipelineResource &input_resource =
      pipeline.resources[pipeline.residency_input];
  PipelineResource &output_resource =
      pipeline.resources[pipeline.residency_output];
  if (!input_resource.owned || !output_resource.owned ||
      input_resource.buffer == nullptr || output_resource.buffer == nullptr ||
      input_resource.bytes < projection.input_arena_bytes ||
      output_resource.bytes != projection.output_arena_bytes ||
      output_resource.output == PipelineResource::no_output) {
    return false;
  }
  std::byte *const input = projection.input_host_banks[0];
  std::byte *const output = projection.output_host_banks[0];
  if (input == nullptr || output == nullptr) {
    return false;
  }
  projection.upload = PipelineFrameUpload{
      .buffer = input_resource.buffer.get(),
      .data = input,
      .bytes = static_cast<std::size_t>(projection.input_arena_bytes),
  };
  projection.download = PipelineFrameDownload{
      .buffer = output_resource.buffer.get(),
      .data = output,
      .bytes = static_cast<std::size_t>(projection.output_arena_bytes),
      .output = output_resource.output,
  };
  return true;
}

} // namespace rund::compute::detail

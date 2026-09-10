#include "internal.hpp"

#include <algorithm>
#include <utility>

namespace rund::compute::detail::virtual_graph_prepare_detail {

Status compile_graph_slices(GraphPreparationDraft &draft) noexcept {
  auto sliced =
      draft.graph_reduction
          ? graph_compile::compile_tiled_graph_slices(draft.program)
          : graph_compile::compile_tiled_graph_pointwise_slices(draft.program);
  if (!sliced) {
    return Status::fail(sliced.reason());
  }
  if (sliced->capacity != draft.geometry.input_frame_elements ||
      sliced->stages.size() < 2u ||
      sliced->input_resources.size() != draft.inputs.size() ||
      sliced->stages.back().outputs.size() != 1u ||
      sliced->stages.back().outputs.front() != sliced->output_resource ||
      sliced->stages.back().tile_partial != draft.graph_reduction ||
      std::any_of(sliced->stages.begin(), sliced->stages.end(),
                  [&draft](const graph_compile::TiledGraphStageSlice &stage) {
                    return stage.program == nullptr ||
                           stage.program->device != draft.program->device;
                  })) {
    return Status::fail(Reason::PipelineInvalid);
  }
  for (const GraphPageMapEntry entry : draft.page_map.entries) {
    if (entry.input >= sliced->input_resources.size()) {
      return Status::fail(Reason::PipelineInvalid);
    }
  }
  const auto terminal_input = std::find_if(
      sliced->resources.begin(), sliced->resources.end(),
      [&sliced](const graph_compile::TiledGraphSliceResource &resource) {
        return resource.resource == sliced->stages.back().inputs.front();
      });
  if (terminal_input == sliced->resources.end() ||
      terminal_input->type != draft.input->type ||
      terminal_input->format != draft.input->format ||
      terminal_input->count != draft.geometry.input_frame_elements) {
    return Status::fail(Reason::PipelineInvalid);
  }
  draft.terminal_input = *terminal_input;
  draft.sliced = std::move(sliced).value();
  return Status::success();
}

} // namespace rund::compute::detail::virtual_graph_prepare_detail

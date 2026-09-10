#include "internal.hpp"

#include <new>
#include <utility>
#include <vector>

namespace rund::compute::detail::graph_compile {

Result<TiledGraphSlices> compile_tiled_graph_pointwise_slices(
    const std::shared_ptr<ProgramState> &program) {
  const auto source = slice_detail::inspect_pointwise(program);
  if (!source) {
    return slice_detail::SliceResult::fail(source.reason());
  }
  try {
    TiledGraphSlices result{};
    result.capacity = source->capacity;
    result.input_resources.assign(source->graph->inputs.begin(),
                                  source->graph->inputs.end());
    result.output_resource = source->graph->outputs.front();
    for (std::size_t index = 0u; index < source->graph->steps.size(); ++index) {
      std::vector<std::uint32_t> stage_inputs;
      std::vector<std::uint32_t> stage_outputs;
      auto stage = slice_detail::compile_map_stage(*source, index, stage_inputs,
                                                   stage_outputs);
      if (!stage) {
        return slice_detail::SliceResult::fail(stage.reason());
      }
      result.stages.push_back(TiledGraphStageSlice{
          .program = std::move(stage).value(),
          .inputs = std::move(stage_inputs),
          .outputs = std::move(stage_outputs),
          .node = static_cast<std::uint32_t>(index),
      });
    }
    slice_detail::append_referenced_resources(result, *source->graph);
    return slice_detail::SliceResult::success(std::move(result));
  } catch (const std::bad_alloc &) {
    return slice_detail::SliceResult::fail(Reason::ProgramCapacity);
  }
}

} // namespace rund::compute::detail::graph_compile

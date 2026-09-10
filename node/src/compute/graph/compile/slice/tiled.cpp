#include "internal.hpp"

#include "semantic.hpp"

#include <new>
#include <span>
#include <utility>
#include <vector>

namespace rund::compute::detail::graph_compile {

Result<TiledGraphSlices>
compile_tiled_graph_slices(const std::shared_ptr<ProgramState> &program) {
  const auto source = slice_detail::inspect_reduce(program);
  if (!source) {
    return slice_detail::SliceResult::fail(source.reason());
  }
  try {
    TiledGraphSlices result{};
    result.capacity = source->capacity;
    result.input_resources.assign(source->graph->inputs.begin(),
                                  source->graph->inputs.end());
    result.output_resource = source->graph->outputs.front();
    if (slice_detail::requires_external_wavefront(*source)) {
      for (std::size_t index = 0u; index + 1u < source->graph->steps.size();
           ++index) {
        std::vector<std::uint32_t> stage_inputs;
        std::vector<std::uint32_t> stage_outputs;
        auto stage = slice_detail::compile_map_stage(
            *source, index, stage_inputs, stage_outputs);
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
      const std::span<const std::uint32_t> terminal_inputs =
          source->graph->value_ids.view(source->reduce->inputs);
      if (terminal_inputs.size() != 1u) {
        return slice_detail::SliceResult::fail(Reason::GraphBindingInvalid);
      }
      auto semantic = compile_service_free_map_semantic_inputs(
          *source->graph, source->type, source->capacity, source->graph->inputs,
          terminal_inputs.front());
      if (!semantic) {
        return slice_detail::SliceResult::fail(semantic.reason());
      }
      result.service_free_prefix = std::move(semantic).value();
    } else {
      std::vector<std::uint32_t> prefix_inputs;
      std::vector<std::uint32_t> prefix_outputs;
      auto prefix = slice_detail::compile_map_prefix(*source, prefix_inputs,
                                                     prefix_outputs);
      if (!prefix) {
        return slice_detail::SliceResult::fail(prefix.reason());
      }
      result.stages.push_back(TiledGraphStageSlice{
          .program = std::move(prefix).value(),
          .inputs = std::move(prefix_inputs),
          .outputs = std::move(prefix_outputs),
          .node = static_cast<std::uint32_t>(source->graph->steps.size() - 2u),
      });
    }
    auto reduce = slice_detail::compile_reduce(*source);
    if (!reduce) {
      return slice_detail::SliceResult::fail(reduce.reason());
    }
    const std::span<const std::uint32_t> reduce_inputs =
        source->graph->value_ids.view(source->reduce->inputs);
    const std::span<const std::uint32_t> reduce_outputs =
        source->graph->value_ids.view(source->reduce->outputs);
    result.stages.push_back(TiledGraphStageSlice{
        .program = std::move(reduce).value(),
        .inputs = std::vector<std::uint32_t>{reduce_inputs.begin(),
                                             reduce_inputs.end()},
        .outputs = std::vector<std::uint32_t>{reduce_outputs.begin(),
                                              reduce_outputs.end()},
        .node = static_cast<std::uint32_t>(source->graph->steps.size() - 1u),
        .tile_partial = true,
    });
    slice_detail::append_referenced_resources(result, *source->graph);
    return slice_detail::SliceResult::success(std::move(result));
  } catch (const std::bad_alloc &) {
    return slice_detail::SliceResult::fail(Reason::ProgramCapacity);
  }
}

} // namespace rund::compute::detail::graph_compile

#include "local.hpp"

#include "../../program/cache.hpp"
#include "../describe.hpp"

#include <span>
#include <utility>

namespace rund::compute::detail {
namespace {

[[nodiscard]] Result<std::shared_ptr<ProgramState>>
compile_program(const std::shared_ptr<GraphState> &graph,
                const graph::Info &layout,
                std::vector<compute_dsl::ComputeOp> operations) {
  auto program = graph_compile::prepare(graph, layout);
  if (!program) {
    return program;
  }
  graph_compile::Lowering lowering{
      .graph = graph,
      .layout = &layout,
      .program = std::move(program).value(),
      .operations = std::move(operations),
  };
  const Status lowered = graph_compile::lower(lowering);
  return lowered ? graph_compile::finish(std::move(lowering))
                 : Result<std::shared_ptr<ProgramState>>::fail(
                       lowered.reason());
}

} // namespace

Result<std::shared_ptr<ProgramState>>
compile_graph(const std::shared_ptr<GraphState> &graph,
              const std::span<const Type> inputs,
              const std::span<const Type> outputs,
              const std::shared_ptr<ProgramCacheState> &cache) {
  if (graph == nullptr) {
    return Result<std::shared_ptr<ProgramState>>::fail(Reason::GraphCapacity);
  }
  if (!graph->status) {
    return Result<std::shared_ptr<ProgramState>>::fail(graph->status.reason());
  }
  if (graph->inputs.empty() || graph->inputs.size() != inputs.size() ||
      graph->outputs.empty() || graph->outputs.size() != outputs.size() ||
      graph->steps.empty()) {
    return Result<std::shared_ptr<ProgramState>>::fail(
        Reason::GraphIncomplete);
  }
  for (std::size_t index = 0u; index < inputs.size(); ++index) {
    if (graph->values[graph->inputs[index] - 1u].type != inputs[index]) {
      return Result<std::shared_ptr<ProgramState>>::fail(
          Reason::GraphTypeMismatch);
    }
  }
  for (std::size_t index = 0u; index < outputs.size(); ++index) {
    if (graph->values[graph->outputs[index] - 1u].type != outputs[index]) {
      return Result<std::shared_ptr<ProgramState>>::fail(
          Reason::GraphTypeMismatch);
    }
  }
  graph_detail::Description description = graph_detail::describe(graph);
  if (!description.status) {
    return Result<std::shared_ptr<ProgramState>>::fail(
        description.status.reason());
  }
  if (cache != nullptr && cache->device != graph->device) {
    return Result<std::shared_ptr<ProgramState>>::fail(
        Reason::ProgramCacheDeviceMismatch);
  }
  const auto build = [&]() -> Result<std::shared_ptr<ProgramState>> {
    auto compiled = compile_program(
        graph, description.info, std::move(description.map_operations));
    if (!compiled) {
      return compiled;
    }
    (*compiled)->graph_info = std::move(description.info);
    return compiled;
  };
  return cache == nullptr
             ? build()
             : cached_program(cache, description.info.fingerprint, build);
}

} // namespace rund::compute::detail

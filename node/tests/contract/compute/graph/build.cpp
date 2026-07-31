#include "../../../../src/compute/graph/build/index.hpp"
#include "../../../../src/compute/graph/state.hpp"

#include <rund/compute/abi/graph.hpp>

#include <array>
#include <cstdint>
#include <memory>
#include <variant>
#include <vector>

namespace {

[[nodiscard]] bool exact_index_handles_full_collision_chain() {
  using Index = rund::compute::detail::graph_build_detail::Index<
      std::uint32_t, rund::compute::detail::MaxOutputs>;
  Index index;
  for (std::size_t ordinal = 0u; ordinal < rund::compute::detail::MaxOutputs;
       ++ordinal) {
    const std::uint32_t key = static_cast<std::uint32_t>(1u + ordinal * 32u);
    const auto [found, inserted] = index.admit(key, ordinal);
    if (!inserted || found != ordinal) {
      return false;
    }
  }
  for (std::size_t ordinal = 0u; ordinal < rund::compute::detail::MaxOutputs;
       ++ordinal) {
    const std::uint32_t key = static_cast<std::uint32_t>(1u + ordinal * 32u);
    if (index.find(key) != ordinal) {
      return false;
    }
    const auto [found, inserted] = index.admit(key, 0u);
    if (inserted || found != ordinal) {
      return false;
    }
  }
  return !index.find(2u);
}

[[nodiscard]] bool duplicate_external_output_keeps_identity() {
  using namespace rund::compute;
  auto graph = std::make_shared<detail::GraphState>();
  graph->count = 4u;
  const std::uint32_t input =
      detail::graph_input_count(graph, detail::Type::I32, graph->count);
  const std::array selected{input, input};
  detail::graph_outputs(graph, selected);
  if (!graph->status || graph->inputs != std::vector<std::uint32_t>{input} ||
      graph->outputs.size() != 1u || graph->outputs.front() == input ||
      graph->identity_outputs !=
          std::vector<std::uint32_t>{graph->outputs.front(),
                                     graph->outputs.front()} ||
      graph->values.size() != 2u || graph->steps.size() != 1u ||
      !std::holds_alternative<detail::MapStep>(graph->steps.front())) {
    return false;
  }

  const std::uint32_t output = graph->outputs.front();
  const std::array direct{output};
  detail::graph_identity_outputs(graph, direct);
  if (!graph->status || !graph->identity_outputs.empty()) {
    return false;
  }
  const std::array invalid{input};
  detail::graph_identity_outputs(graph, invalid);
  return !graph->status && graph->status.reason() == Reason::GraphValueInvalid;
}

} // namespace

int RunComputeGraphBuildContract() {
  return exact_index_handles_full_collision_chain() &&
                 duplicate_external_output_keeps_identity()
             ? 0
             : 1;
}

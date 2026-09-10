#include "local.hpp"

#include "src/compute/cpu/graph.hpp"
#include "src/compute/job/view.hpp"
#include "src/compute/program/state.hpp"

#include <array>
#include <cstdint>
#include <vector>

namespace rund_node_memory_contract::graph_detail {

int CheckViewRequirements(
    const std::shared_ptr<rund::compute::detail::ProgramState> &program) {
  using namespace rund::compute;
  using namespace rund::compute::detail;
  const auto first = plan_cpu_view_transfer_requirements(program);
  const auto second = plan_cpu_view_transfer_requirements(program);
  if (!first || !second || *first != *second ||
      first->program != program.get() || !first->inputs.empty() ||
      first->outputs != std::vector<std::uint32_t>{0u}) {
    return 1;
  }
  constexpr std::array<JobBufferView, 1u> strided_input{{
      {.count = 4u, .stride = 2u, .element_bytes = 4u, .alignment = 4u},
  }};
  constexpr std::array<JobBufferView, 1u> strided_output{{
      {.offset = 1u,
       .count = 4u,
       .stride = 3u,
       .element_bytes = 4u,
       .alignment = 4u},
  }};
  const auto first_route =
      plan_cpu_view_transfers(program, strided_input, strided_output, &*first);
  constexpr std::array<JobBufferView, 1u> dense_input{{
      {.count = 4u, .stride = 1u, .element_bytes = 4u, .alignment = 4u},
  }};
  constexpr std::array<JobBufferView, 1u> dense_output{{
      {.count = 4u, .stride = 1u, .element_bytes = 4u, .alignment = 4u},
  }};
  const auto second_route =
      plan_cpu_view_transfers(program, dense_input, dense_output, &*first);
  if (!first_route || !second_route || !first_route->inputs.empty() ||
      first_route->outputs.size() != 1u || first_route->bytes != 16u ||
      !second_route->inputs.empty() || !second_route->outputs.empty() ||
      second_route->bytes != 0u) {
    return 2;
  }
  CpuViewTransferRequirements forged = *first;
  forged.outputs.push_back(0u);
  const auto duplicate =
      plan_cpu_view_transfers(program, strided_input, strided_output, &forged);
  forged = *first;
  forged.outputs.front() = 1u;
  const auto out_of_range =
      plan_cpu_view_transfers(program, strided_input, strided_output, &forged);
  forged = *first;
  ++forged.graph_hash;
  const auto wrong_graph =
      plan_cpu_view_transfers(program, strided_input, strided_output, &forged);
  return !duplicate && !out_of_range && !wrong_graph &&
                 duplicate.reason() == Reason::PipelineInvalid &&
                 out_of_range.reason() == Reason::PipelineInvalid &&
                 wrong_graph.reason() == Reason::PipelineInvalid
             ? 0
             : 3;
}

} // namespace rund_node_memory_contract::graph_detail

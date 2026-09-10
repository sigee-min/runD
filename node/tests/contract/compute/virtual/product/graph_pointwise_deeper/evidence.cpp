#include "internal.hpp"

#include "../graph_pointwise_shape/evidence.hpp"

#include <span>

namespace rund_node_test_virtual::product::graph_pointwise_deeper {

std::array<std::uint64_t, StageCount>
stage_generations(const Case &test_case) noexcept {
  std::array<std::uint64_t, StageCount> result{};
  graph_pointwise_shape::stage_generations(test_case.state, result);
  return result;
}

bool validate_case(
    Case &test_case, const rund::compute::Backend backend,
    const rund::compute::Status &status, const std::uint64_t initial_version,
    const std::array<std::uint64_t, StageCount> &initial_generations) noexcept {
  return graph_pointwise_shape::validate(
      graph_pointwise_shape::EvidenceView{
          .label = "deeper",
          .input_count = InputCount,
          .stage_count = StageCount,
          .frame_elements = FrameElements,
          .page_count = PageCount,
          .tail_elements = TailElements,
          .frame_capacity = test_case.pipeline.plan().residency.frame_capacity,
          .stats = test_case.pipeline.stats(),
          .state = test_case.state,
          .input_backings = std::span{test_case.input_backings},
          .output_backing = test_case.output_backing,
          .expected = std::span{test_case.expected},
      },
      backend, status, initial_version, initial_generations);
}

} // namespace rund_node_test_virtual::product::graph_pointwise_deeper

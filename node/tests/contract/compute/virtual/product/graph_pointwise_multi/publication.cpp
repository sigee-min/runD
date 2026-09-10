#include "internal.hpp"

#include "src/compute/device/residency/pool.hpp"
#include "src/compute/virtual/state.hpp"

#include <limits>
#include <mutex>

namespace rund_node_test_virtual::product::graph_pointwise_multi {

std::array<std::uint64_t, StageCount>
stage_generations(const Case &test_case) noexcept {
  std::array<std::uint64_t, StageCount> generations{};
  if (test_case.state == nullptr ||
      test_case.state->graph_pipelines.size() !=
          StageCount * rund::compute::detail::residency::Pool::BankCount) {
    generations.fill(std::numeric_limits<std::uint64_t>::max());
    return generations;
  }
  for (std::size_t stage = 0u; stage < StageCount; ++stage) {
    const auto &pipeline =
        test_case.state->graph_pipelines
            [stage * rund::compute::detail::residency::Pool::BankCount];
    if (pipeline == nullptr || pipeline->publication == nullptr) {
      generations.fill(std::numeric_limits<std::uint64_t>::max());
      return generations;
    }
    std::lock_guard state_lock{pipeline->gate};
    std::lock_guard publication_lock{pipeline->publication->gate};
    generations[stage] = pipeline->publication->generation;
  }
  return generations;
}

} // namespace rund_node_test_virtual::product::graph_pointwise_multi

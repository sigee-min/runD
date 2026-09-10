#include "local.hpp"

#include "../../../../../../src/compute/resource/index.hpp"

#include <array>
#include <cstdint>
#include <vector>

namespace rund_node_graph_services::resource_test_detail {

using rund::compute::resource::Access;
using rund::compute::resource::AccessMode;
using rund::compute::resource::Resource;

[[nodiscard]] std::uint32_t reverse_bits(std::uint32_t value,
                                         const std::uint32_t width) noexcept {
  std::uint32_t reversed = 0u;
  for (std::uint32_t bit = 0u; bit < width; ++bit) {
    reversed = (reversed << 1u) | (value & 1u);
    value >>= 1u;
  }
  return reversed;
}

[[nodiscard]] bool bounded_disjoint_work() {
  using rund::compute::resource::detail::AnalysisStats;
  using rund::compute::resource::detail::analyze_measured;
  constexpr std::uint32_t Width = 12u;
  constexpr std::uint32_t Count = 1u << Width;
  const std::array<Resource, 1u> resources{Resource{
      .id = 1u,
      .bytes = static_cast<std::uint64_t>(Count) * 2u,
      .alias_group = 47u,
      .alias_offset_bytes = 0u,
  }};
  std::vector<Access> accesses;
  accesses.reserve(Count);
  for (std::uint32_t index = 0u; index < Count; ++index) {
    accesses.push_back(Access{
        .node = index,
        .resource = 1u,
        .mode = AccessMode::Write,
        .offset_bytes =
            static_cast<std::uint64_t>(reverse_bits(index, Width)) * 2u,
        .size_bytes = 1u,
    });
  }
  AnalysisStats stats;
  const auto plan = analyze_measured(resources, accesses, Count, stats);
  constexpr std::uint64_t LogBound = Width + 1u;
  constexpr std::uint64_t IndexedBound =
      4u * static_cast<std::uint64_t>(Count) * LogBound;
  constexpr std::uint64_t BrutePairs =
      static_cast<std::uint64_t>(Count) * (Count - 1u) / 2u;
  const std::uint64_t indexed_visits = stats.insert_visits + stats.query_visits;
  return plan && plan->dependencies.empty() && plan->barriers.empty() &&
         stats.envelope_candidates == 0u && stats.exact_checks == 0u &&
         indexed_visits <= IndexedBound && indexed_visits * 32u < BrutePairs;
}

[[nodiscard]] bool bounded_frontier_work() {
  using rund::compute::resource::detail::AnalysisStats;
  using rund::compute::resource::detail::analyze_measured;
  constexpr std::uint32_t Width = 12u;
  constexpr std::uint32_t Count = 1u << Width;
  const std::array<Resource, 1u> resources{Resource{
      .id = 1u,
      .bytes = 64u,
      .alias_group = 53u,
      .alias_offset_bytes = 0u,
  }};
  std::vector<Access> accesses;
  accesses.reserve(Count);
  for (std::uint32_t index = 0u; index < Count; ++index) {
    accesses.push_back(Access{
        .node = index,
        .resource = 1u,
        .mode = AccessMode::Write,
        .offset_bytes = 0u,
        .size_bytes = 64u,
    });
  }
  AnalysisStats stats;
  const auto plan = analyze_measured(resources, accesses, Count, stats);
  constexpr std::uint64_t LogBound = Width + 1u;
  constexpr std::uint64_t IndexedBound =
      8u * static_cast<std::uint64_t>(Count) * LogBound;
  const std::uint64_t indexed_visits = stats.insert_visits + stats.query_visits;
  return plan && plan->dependencies.size() == Count - 1u &&
         plan->barriers.size() == Count - 1u &&
         stats.envelope_candidates == Count - 1u &&
         stats.exact_checks == Count - 1u && indexed_visits <= IndexedBound;
}

} // namespace rund_node_graph_services::resource_test_detail

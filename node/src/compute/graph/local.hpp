#pragma once

#include "../status.hpp"
#include "state.hpp"

#include <kernel/program/compute/limit.hpp>

#include <optional>

namespace rund::compute::detail::graph_detail {

inline constexpr std::size_t MaxValues =
    static_cast<std::size_t>(kernel::kMaxGraphValueCount);
inline constexpr std::size_t MaxSteps =
    static_cast<std::size_t>(kernel::kMaxGraphNodeCount);

[[nodiscard]] inline bool valid(const GraphState &graph,
                                const std::uint32_t value) noexcept {
  return value > 0u && value <= graph.values.size();
}

inline void reject(GraphState &graph, const Reason reason) {
  if (graph.status) {
    graph.status = Status::fail(reason);
  }
}

[[nodiscard]] inline std::uint32_t append(GraphState &graph, const Type type,
                                          const std::size_t count,
                                          const FixedFormat fixed_format = {}) {
  if (!graph.status) {
    return 0u;
  }
  if (graph.values.size() >= MaxValues) {
    reject(graph, Reason::GraphCapacity);
    return 0u;
  }
  try {
    graph.values.push_back(
        GraphValue{.type = type, .fixed_format = fixed_format, .count = count});
  } catch (const std::bad_alloc &) {
    reject(graph, Reason::GraphCapacity);
    return 0u;
  }
  return static_cast<std::uint32_t>(graph.values.size());
}

[[nodiscard]] inline std::optional<kernel::ScanOp>
scan_operation(const Scan operation) noexcept {
  switch (operation) {
  case Scan::ExclusiveSum:
    return kernel::ScanOp::ExclusiveSum;
  case Scan::InclusiveSum:
    return kernel::ScanOp::InclusiveSum;
  }
  return std::nullopt;
}

[[nodiscard]] inline constexpr std::uint64_t
collective_block(const std::uint64_t count) noexcept {
  std::uint64_t block = 1u;
  while (block < count && block < 256u) {
    block <<= 1u;
  }
  return block;
}

} // namespace rund::compute::detail::graph_detail

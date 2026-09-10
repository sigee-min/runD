#pragma once

#include <cstddef>
#include <cstdint>

namespace rund_node_test_virtual::product::graph_pointwise_shape {

template <std::uint64_t First, std::size_t Count, class Expression>
[[nodiscard]] constexpr auto add_stage(Expression value) {
  static_assert(Count != 0u);
  if constexpr (Count == 1u) {
    return value + First;
  } else {
    constexpr std::size_t Left = Count / 2u;
    return add_stage<First, Left>(value) +
           add_stage<First + Left, Count - Left>(value);
  }
}

template <std::size_t Count>
[[nodiscard]] constexpr std::uint64_t
stage_value(const std::uint64_t value, const std::uint64_t first) noexcept {
  static_assert(Count != 0u);
  constexpr std::uint64_t LiteralSum = Count * (Count + 1u) / 2u;
  return value * Count + LiteralSum + (first - 1u) * Count;
}

} // namespace rund_node_test_virtual::product::graph_pointwise_shape

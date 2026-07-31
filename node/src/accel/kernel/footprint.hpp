#pragma once

#include <cstdint>
#include <limits>
#include <vector>

namespace rund::node::accel::detail {

template <class T>
[[nodiscard]] inline std::uint64_t
capacity_bytes(const std::vector<T> &values) noexcept {
  constexpr std::uint64_t maximum = std::numeric_limits<std::uint64_t>::max();
  return values.capacity() > maximum / sizeof(T)
             ? maximum
             : static_cast<std::uint64_t>(values.capacity()) * sizeof(T);
}

} // namespace rund::node::accel::detail

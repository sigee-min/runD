#pragma once

#include <kernel/program/skeleton/shape.hpp>

#include <array>
#include <cstddef>
#include <type_traits>

namespace rund::kernel {

template <typename T, std::size_t Rank>
  requires(Rank > 0u)
struct View {
  T* data = nullptr;
  Index<Rank> shape{};
  std::array<std::ptrdiff_t, Rank> stride{};
  u64 element_count = 0u;
  std::size_t alignment_bytes = 0u;
  ViewAccessPattern access = ViewAccessPattern::Unsupported;
  bool valid = false;
  const char* reason = "view_not_validated";

  [[nodiscard]] constexpr explicit operator bool() const noexcept { return valid; }

  [[nodiscard]] constexpr T& operator()(const Index<Rank>& index) const noexcept {
    std::ptrdiff_t offset = 0;
    for (std::size_t axis = 0u; axis < Rank; ++axis) {
      offset += static_cast<std::ptrdiff_t>(index[axis]) * stride[axis];
    }
    return data[offset];
  }

  template <typename... Coordinates>
    requires(sizeof...(Coordinates) == Rank &&
             (std::is_integral_v<Coordinates> && ...))
  [[nodiscard]] constexpr T& operator()(Coordinates... coordinates) const noexcept {
    return (*this)(Index<Rank>{static_cast<u64>(coordinates)...});
  }
};

} // namespace rund::kernel

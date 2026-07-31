#pragma once

#include <kernel/program/skeleton/shape.hpp>
#include <kernel/program/skeleton/model.hpp>

#include <limits>
#include <type_traits>

namespace rund::kernel {

template <typename... Extents>
  requires(sizeof...(Extents) > 0u && (std::is_integral_v<Extents> && ...))
[[nodiscard]] constexpr auto space(Extents... extents) noexcept
    -> Space<sizeof...(Extents)> {
  if ((skeleton_detail::Negative(extents) || ...)) {
    return Space<sizeof...(Extents)>{
        .valid = false,
        .reason = "skeleton_negative_extent",
    };
  }
  return Space<sizeof...(Extents)>{
      .extent = Index<sizeof...(Extents)>{static_cast<u64>(extents)...},
  };
}

template <std::size_t Rank>
  requires(Rank > 0u)
[[nodiscard]] constexpr Space<Rank> space(const Index<Rank>& extent) noexcept {
  return Space<Rank>{.extent = extent};
}

template <typename Units>
  requires std::is_integral_v<Units>
[[nodiscard]] constexpr Alignment align(const Units units) noexcept {
  if (skeleton_detail::Negative(units)) {
    return Alignment{
        .valid = false,
        .reason = "skeleton_alignment_negative",
    };
  }
  const u64 unsigned_units = static_cast<u64>(units);
  if (unsigned_units == 0u) {
    return Alignment{
        .valid = false,
        .reason = "skeleton_alignment_zero",
    };
  }
  if (unsigned_units > std::numeric_limits<u32>::max()) {
    return Alignment{
        .valid = false,
        .reason = "skeleton_alignment_exceeds_u32",
    };
  }
  return Alignment{.units = static_cast<u32>(unsigned_units)};
}

} // namespace rund::kernel

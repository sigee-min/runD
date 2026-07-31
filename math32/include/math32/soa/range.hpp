#pragma once

#include <cstdint>
#include <span>

namespace rund::math32::soa::detail {

template <typename Lhs, typename Rhs>
[[nodiscard]] inline bool SameRange(const std::span<Lhs> lhs,
                                    const std::span<Rhs> rhs) noexcept {
  return reinterpret_cast<std::uintptr_t>(lhs.data()) ==
             reinterpret_cast<std::uintptr_t>(rhs.data()) &&
         lhs.size_bytes() == rhs.size_bytes();
}

template <typename Lhs, typename Rhs>
[[nodiscard]] inline bool Overlaps(const std::span<Lhs> lhs,
                                   const std::span<Rhs> rhs) noexcept {
  if (lhs.empty() || rhs.empty()) {
    return false;
  }
  const auto lhs_begin = reinterpret_cast<std::uintptr_t>(lhs.data());
  const auto rhs_begin = reinterpret_cast<std::uintptr_t>(rhs.data());
  return lhs_begin <= rhs_begin
             ? rhs_begin - lhs_begin < lhs.size_bytes()
             : lhs_begin - rhs_begin < rhs.size_bytes();
}

template <typename Lhs, typename Rhs>
[[nodiscard]] inline bool PartiallyOverlaps(const std::span<Lhs> lhs,
                                            const std::span<Rhs> rhs) noexcept {
  return Overlaps(lhs, rhs) && !SameRange(lhs, rhs);
}

}  // namespace rund::math32::soa::detail

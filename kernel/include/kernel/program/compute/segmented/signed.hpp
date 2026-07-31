#pragma once

#include <kernel/core/model.hpp>

#include <concepts>
#include <limits>
#include <type_traits>

namespace rund::kernel {

template <class Value>
concept SignedLane =
    std::signed_integral<Value> &&
    (sizeof(Value) == sizeof(i32) || sizeof(Value) == sizeof(i64));

namespace segmented_signed_detail {

template <SignedLane Value>
using Prefix =
    std::conditional_t<sizeof(Value) == sizeof(i32), i64, __int128_t>;

template <SignedLane Value> using Total = __int128_t;

template <SignedLane Value, class Wide>
[[nodiscard]] constexpr bool Fits(const Wide value) noexcept {
  return value >= static_cast<Wide>(std::numeric_limits<Value>::min()) &&
         value <= static_cast<Wide>(std::numeric_limits<Value>::max());
}

template <SignedLane Value, class Wide>
[[nodiscard]] constexpr u64 Bits(const Wide value) noexcept {
  using Word = std::make_unsigned_t<Value>;
  return static_cast<u64>(static_cast<Word>(value));
}

} // namespace segmented_signed_detail
} // namespace rund::kernel

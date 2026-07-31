#pragma once

#include <rund/compute/expr/functions/approx.hpp>

#include <bit>
#include <cstdint>
#include <limits>
#include <type_traits>

namespace rund::compute {

struct HashOp final {
  struct UnitTag final {};

  inline static constexpr UnitTag Unit{};
};

namespace detail {

template <class T>
[[nodiscard]] constexpr T raw_value(const std::uint64_t bits) noexcept {
  if constexpr (FixedValue<T>) {
    using Raw = typename T::Raw;
    using Unsigned = std::make_unsigned_t<Raw>;
    return T::from_raw(std::bit_cast<Raw>(static_cast<Unsigned>(bits)));
  } else if constexpr (std::signed_integral<T>) {
    using Unsigned = std::make_unsigned_t<T>;
    return std::bit_cast<T>(static_cast<Unsigned>(bits));
  } else {
    return static_cast<T>(bits);
  }
}

template <class E>
[[nodiscard]] constexpr auto raw_like(const E &anchor,
                                      const std::uint64_t bits) {
  using T = FunctionExprValueT<E>;
  return literal_like(anchor, raw_value<T>(bits));
}

template <class E>
  requires FixedExpression<E>
[[nodiscard]] constexpr auto fractional_mask_like(const E &anchor) {
  using T = FunctionExprValueT<E>;
  static_assert(T::fraction_bits > 0u && T::fraction_bits < 64u);
  constexpr std::uint64_t mask =
      (std::uint64_t{1} << T::fraction_bits) - std::uint64_t{1};
  return raw_like(anchor, mask);
}

template <class E> [[nodiscard]] constexpr auto stored(const E &value) {
  if constexpr (FixedExpression<E>) {
    return storage(value);
  } else {
    return value;
  }
}

template <std::uint32_t FirstShift, std::uint32_t SecondShift,
          std::uint32_t FinalShift>
[[nodiscard]] constexpr auto
hash_finalizer(const auto &value, const std::uint64_t first_multiplier,
               const std::uint64_t second_multiplier) {
  const auto first = bit_xor(value, shr_logical<FirstShift>(value));
  const auto first_product = mul_wrap(first, raw_like(value, first_multiplier));
  const auto second =
      bit_xor(first_product, shr_logical<SecondShift>(first_product));
  const auto second_product =
      mul_wrap(second, raw_like(value, second_multiplier));
  return bit_xor(second_product, shr_logical<FinalShift>(second_product));
}

} // namespace detail

template <class E>
  requires detail::StoredExprValue<detail::FunctionExprValueT<E>>
[[nodiscard]] constexpr auto hash(const E &value) {
  if constexpr (sizeof(detail::FunctionExprValueT<E>) == 8u) {
    return detail::hash_finalizer<33u, 33u, 33u>(value, 0xff51afd7ed558ccdull,
                                                 0xc4ceb9fe1a85ec53ull);
  } else {
    return detail::hash_finalizer<16u, 15u, 16u>(value, 0x7feb352dull,
                                                 0x846ca68bull);
  }
}

[[nodiscard]] constexpr auto hash(const auto &value, const auto &seed) {
  return hash(bit_xor(value, seed));
}

template <class E>
  requires detail::StoredExprValue<detail::FunctionExprValueT<E>>
[[nodiscard]] constexpr auto hash(HashOp::UnitTag, const E &value) {
  if constexpr (detail::FixedExpression<E>) {
    return bit_and(hash(value), detail::fractional_mask_like(value));
  } else {
    constexpr std::uint64_t mask = sizeof(detail::FunctionExprValueT<E>) == 8u
                                       ? 0x7fffffffffffffffull
                                       : 0x7fffffffull;
    return bit_and(hash(value), detail::raw_like(value, mask));
  }
}

[[nodiscard]] constexpr auto hash(HashOp::UnitTag, const auto &value,
                                  const auto &seed) {
  using E = std::remove_cvref_t<decltype(value)>;
  if constexpr (detail::FixedExpression<E>) {
    return bit_and(hash(value, seed), detail::fractional_mask_like(value));
  } else {
    constexpr std::uint64_t mask = sizeof(detail::FunctionExprValueT<E>) == 8u
                                       ? 0x7fffffffffffffffull
                                       : 0x7fffffffull;
    return bit_and(hash(value, seed), detail::raw_like(value, mask));
  }
}

[[nodiscard]] constexpr auto noise(const auto &cell, const auto &amount) {
  const auto next = detail::stored(cell + detail::raw_like(cell, 1u));
  return lerp(hash(HashOp::Unit, cell), hash(HashOp::Unit, next), fade(amount));
}

[[nodiscard]] constexpr auto noise(const auto &cell, const auto &amount,
                                   const auto &seed) {
  const auto next = detail::stored(cell + detail::raw_like(cell, 1u));
  return lerp(hash(HashOp::Unit, cell, seed), hash(HashOp::Unit, next, seed),
              fade(amount));
}

[[nodiscard]] constexpr auto noise(const auto &x, const auto &y, const auto &tx,
                                   const auto &ty, const auto &seed) {
  const auto x_next = detail::stored(x + detail::raw_like(x, 1u));
  const auto y_next = detail::stored(y + detail::raw_like(y, 1u));
  const auto y_seed = bit_xor(y, seed);
  const auto y_next_seed = bit_xor(y_next, seed);
  const auto row0 = lerp(hash(HashOp::Unit, x, y_seed),
                         hash(HashOp::Unit, x_next, y_seed), fade(tx));
  const auto row1 = lerp(hash(HashOp::Unit, x, y_next_seed),
                         hash(HashOp::Unit, x_next, y_next_seed), fade(tx));
  return lerp(row0, row1, fade(ty));
}

[[nodiscard]] constexpr auto noise(const auto &x, const auto &y, const auto &tx,
                                   const auto &ty) {
  return noise(x, y, tx, ty, fixed_zero(x));
}

} // namespace rund::compute

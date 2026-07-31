#pragma once

#include <kernel/program/compute/dsl/expression/value.hpp>

#include <type_traits>

namespace rund::compute_dsl::detail {

template <typename T>
concept ConstantLiteral = std::is_integral_v<CleanType<T>>;

template <ConstantLiteral T>
[[nodiscard]] constexpr rund::kernel::u64
EncodeConstantBits(const ScalarMode mode, const T value) noexcept {
  using Clean = CleanType<T>;
  using Unsigned = std::make_unsigned_t<Clean>;
  if (WideMode(mode)) {
    if constexpr (std::is_signed_v<Clean>) {
      return static_cast<rund::kernel::u64>(
          static_cast<rund::kernel::i64>(value));
    }
    return static_cast<rund::kernel::u64>(static_cast<Unsigned>(value));
  }
  return static_cast<rund::kernel::u32>(static_cast<Unsigned>(value));
}

template <ConstantLiteral T>
[[nodiscard]] inline Expr ConstantValue(const Expr anchor,
                                        const T value) noexcept {
  return Constant(anchor, EncodeConstantBits(ScalarModeOf(anchor), value));
}

template <ConstantLiteral T>
[[nodiscard]] inline Expr StorageConstantValue(const Expr anchor,
                                               const T value) noexcept {
  return StorageConstant(anchor,
                         EncodeConstantBits(ScalarModeOf(anchor), value));
}

} // namespace rund::compute_dsl::detail

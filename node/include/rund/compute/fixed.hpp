#pragma once

#include <rund/compute/status.hpp>

#include <bit>
#include <compare>
#include <cstdint>
#include <limits>
#include <type_traits>

namespace rund::compute {

enum class Rounding : unsigned char {
  TowardZero,
  Down,
  Up,
  NearestEven,
};

enum class Overflow : unsigned char {
  Saturate,
  Wrap,
};

enum class Approximation : unsigned char {
  Exact,
  Deterministic,
};

namespace detail {

struct FixedFormat final {
  unsigned char integer_bits{};
  unsigned char fraction_bits{};
  Rounding rounding{Rounding::NearestEven};
  Overflow overflow{Overflow::Saturate};
  Approximation approximation{Approximation::Exact};

  [[nodiscard]] constexpr bool operator==(const FixedFormat &) const noexcept =
      default;
};

template <unsigned IntegerBits, unsigned FractionBits>
inline constexpr unsigned FixedWidth = IntegerBits + FractionBits;

template <unsigned IntegerBits, unsigned FractionBits>
concept StoredFixedFormat =
    IntegerBits >= 1u && IntegerBits <= 63u && FractionBits >= 1u &&
    FractionBits <= 63u &&
    (FixedWidth<IntegerBits, FractionBits> == 32u ||
     FixedWidth<IntegerBits, FractionBits> == 64u);

template <unsigned Width> struct FixedStorage;
template <> struct FixedStorage<32u> final {
  using Signed = std::int32_t;
  using Unsigned = std::uint32_t;
};
template <> struct FixedStorage<64u> final {
  using Signed = std::int64_t;
  using Unsigned = std::uint64_t;
};

template <class Raw, unsigned FractionBits>
[[nodiscard]] constexpr Raw fixed_ratio_raw(const std::int64_t numerator,
                                            const std::int64_t denominator,
                                            const Rounding rounding) noexcept {
  static_assert(std::is_signed_v<Raw>);
  using Wide = __int128_t;
  using UWide = __uint128_t;
  const Wide signed_numerator = static_cast<Wide>(numerator);
  const Wide signed_denominator = static_cast<Wide>(denominator);
  const bool negative = (signed_numerator < 0) != (signed_denominator < 0);
  const UWide magnitude = signed_numerator < 0
                              ? static_cast<UWide>(-signed_numerator)
                              : static_cast<UWide>(signed_numerator);
  const UWide divisor = signed_denominator < 0
                            ? static_cast<UWide>(-signed_denominator)
                            : static_cast<UWide>(signed_denominator);
  const UWide scaled = magnitude << FractionBits;
  UWide quotient = scaled / divisor;
  const UWide remainder = scaled % divisor;
  const bool nonzero = remainder != 0u;
  const bool nearest = remainder * 2u > divisor ||
                       (remainder * 2u == divisor && (quotient & 1u) != 0u);
  if ((rounding == Rounding::Down && negative && nonzero) ||
      (rounding == Rounding::Up && !negative && nonzero) ||
      (rounding == Rounding::NearestEven && nearest)) {
    ++quotient;
  }
  const Wide value = negative ? -static_cast<Wide>(quotient)
                              : static_cast<Wide>(quotient);
  constexpr Wide low = static_cast<Wide>(std::numeric_limits<Raw>::min());
  constexpr Wide high = static_cast<Wide>(std::numeric_limits<Raw>::max());
  return value < low ? std::numeric_limits<Raw>::min()
                     : value > high ? std::numeric_limits<Raw>::max()
                                    : static_cast<Raw>(value);
}

template <class TargetRaw, Overflow Mode>
[[nodiscard]] constexpr TargetRaw narrow_fixed(const __int128_t value) noexcept {
  if constexpr (Mode == Overflow::Saturate) {
    constexpr auto low = static_cast<__int128_t>(
        std::numeric_limits<TargetRaw>::min());
    constexpr auto high = static_cast<__int128_t>(
        std::numeric_limits<TargetRaw>::max());
    return value < low ? std::numeric_limits<TargetRaw>::min()
                       : value > high ? std::numeric_limits<TargetRaw>::max()
                                      : static_cast<TargetRaw>(value);
  } else {
    using Unsigned = std::make_unsigned_t<TargetRaw>;
    constexpr unsigned width = sizeof(TargetRaw) * 8u;
    const __uint128_t bits = static_cast<__uint128_t>(value);
    const __uint128_t mask = width == 64u
                                 ? static_cast<__uint128_t>(
                                       std::numeric_limits<std::uint64_t>::max())
                                 : ((static_cast<__uint128_t>(1u) << width) - 1u);
    return std::bit_cast<TargetRaw>(static_cast<Unsigned>(bits & mask));
  }
}

template <Rounding Mode>
[[nodiscard]] constexpr __int128_t round_shift_right(
    const __int128_t value, const unsigned shift) noexcept {
  if (shift == 0u) {
    return value;
  }
  using UWide = __uint128_t;
  const bool negative = value < 0;
  const UWide magnitude = negative ? static_cast<UWide>(-value)
                                   : static_cast<UWide>(value);
  UWide quotient = magnitude >> shift;
  const UWide mask = (static_cast<UWide>(1u) << shift) - 1u;
  const UWide remainder = magnitude & mask;
  const UWide halfway = static_cast<UWide>(1u) << (shift - 1u);
  const bool nonzero = remainder != 0u;
  const bool nearest = remainder > halfway ||
                       (remainder == halfway && (quotient & 1u) != 0u);
  if ((Mode == Rounding::Down && negative && nonzero) ||
      (Mode == Rounding::Up && !negative && nonzero) ||
      (Mode == Rounding::NearestEven && nearest)) {
    ++quotient;
  }
  return negative ? -static_cast<__int128_t>(quotient)
                  : static_cast<__int128_t>(quotient);
}

} // namespace detail

template <unsigned IntegerBits, unsigned FractionBits>
  requires detail::StoredFixedFormat<IntegerBits, FractionBits>
class Fixed final {
public:
  static constexpr unsigned integer_bits = IntegerBits;
  static constexpr unsigned fraction_bits = FractionBits;
  static constexpr unsigned storage_bits = IntegerBits + FractionBits;
  using Raw = typename detail::FixedStorage<storage_bits>::Signed;

  constexpr Fixed() noexcept = default;

  [[nodiscard]] static constexpr Fixed from_raw(const Raw value) noexcept {
    return Fixed{value};
  }

  template <class Value>
    requires std::is_floating_point_v<std::remove_cvref_t<Value>>
  [[nodiscard]] static constexpr Fixed from_raw(Value) noexcept = delete;

  [[nodiscard]] static Result<Fixed>
  from_ratio(const std::int64_t numerator, const std::int64_t denominator,
             const Rounding rounding = Rounding::NearestEven) noexcept {
    if (denominator == 0) {
      return Result<Fixed>::fail(Reason::FixedRatioZeroDenominator);
    }
    return Result<Fixed>::success(from_raw(
        detail::fixed_ratio_raw<Raw, FractionBits>(numerator, denominator,
                                                   rounding)));
  }

  template <class Numerator, class Denominator>
    requires(std::is_floating_point_v<std::remove_cvref_t<Numerator>> ||
             std::is_floating_point_v<std::remove_cvref_t<Denominator>>)
  [[nodiscard]] static Result<Fixed>
  from_ratio(Numerator, Denominator,
             Rounding = Rounding::NearestEven) noexcept = delete;

  template <std::int64_t Numerator, std::int64_t Denominator,
            Rounding Mode = Rounding::NearestEven>
  [[nodiscard]] static consteval Fixed ratio() noexcept {
    static_assert(Denominator != 0, "fixed ratio denominator must be nonzero");
    return from_raw(detail::fixed_ratio_raw<Raw, FractionBits>(
        Numerator, Denominator, Mode));
  }

  [[nodiscard]] static constexpr Fixed zero() noexcept { return {}; }
  [[nodiscard]] static constexpr Fixed min() noexcept {
    return from_raw(std::numeric_limits<Raw>::min());
  }
  [[nodiscard]] static constexpr Fixed max() noexcept {
    return from_raw(std::numeric_limits<Raw>::max());
  }
  [[nodiscard]] constexpr Raw raw() const noexcept { return raw_; }
  auto operator<=>(const Fixed &) const = default;

private:
  constexpr explicit Fixed(const Raw value) noexcept : raw_(value) {}
  Raw raw_{};
};

namespace detail {
template <class T> inline constexpr bool IsFixedTarget = false;
template <unsigned I, unsigned F>
inline constexpr bool IsFixedTarget<Fixed<I, F>> = true;
} // namespace detail

template <class Target, Rounding Round = Rounding::NearestEven,
          Overflow OverflowMode = Overflow::Saturate,
          unsigned SourceIntegerBits, unsigned SourceFractionBits>
  requires detail::IsFixedTarget<std::remove_cv_t<Target>> &&
           detail::StoredFixedFormat<Target::integer_bits,
                                     Target::fraction_bits>
[[nodiscard]] constexpr Target
quantize(const Fixed<SourceIntegerBits, SourceFractionBits> value) noexcept {
  using TargetRaw = typename Target::Raw;
  __int128_t scaled = static_cast<__int128_t>(value.raw());
  if constexpr (Target::fraction_bits > SourceFractionBits) {
    constexpr unsigned shift = Target::fraction_bits - SourceFractionBits;
    scaled *= static_cast<__int128_t>(1u) << shift;
  } else if constexpr (Target::fraction_bits < SourceFractionBits) {
    scaled = detail::round_shift_right<Round>(
        scaled, SourceFractionBits - Target::fraction_bits);
  }
  return Target::from_raw(
      detail::narrow_fixed<TargetRaw, OverflowMode>(scaled));
}

} // namespace rund::compute

namespace rund::compute::detail {

enum class Type : unsigned char {
  I32,
  U32,
  I64,
  U64,
  FixedLane32,
  FixedLane64,
};

template <class T> inline constexpr bool IsFixedValue = false;
template <unsigned I, unsigned F>
inline constexpr bool IsFixedValue<Fixed<I, F>> = true;

template <class T>
inline constexpr bool IsComputeValue =
    std::is_same_v<std::remove_cv_t<T>, std::int32_t> ||
    std::is_same_v<std::remove_cv_t<T>, std::uint32_t> ||
    std::is_same_v<std::remove_cv_t<T>, std::int64_t> ||
    std::is_same_v<std::remove_cv_t<T>, std::uint64_t> ||
    IsFixedValue<std::remove_cv_t<T>>;

template <class T>
concept ComputeValue = IsComputeValue<T>;

template <class T>
concept FixedValue = IsFixedValue<std::remove_cv_t<T>>;

template <class> inline constexpr bool UnsupportedType = false;

template <class T>
[[nodiscard]] consteval FixedFormat fixed_format(
    const Rounding rounding = Rounding::NearestEven,
    const Overflow overflow = Overflow::Saturate,
    const Approximation approximation = Approximation::Exact) {
  using Value = std::remove_cv_t<T>;
  static_assert(FixedValue<Value>, "fixed format requires Fixed<I, F>");
  return FixedFormat{
      .integer_bits = static_cast<unsigned char>(Value::integer_bits),
      .fraction_bits = static_cast<unsigned char>(Value::fraction_bits),
      .rounding = rounding,
      .overflow = overflow,
      .approximation = approximation,
  };
}

template <class T> [[nodiscard]] consteval FixedFormat storage_format() {
  if constexpr (FixedValue<std::remove_cv_t<T>>) {
    return fixed_format<std::remove_cv_t<T>>();
  } else {
    return {};
  }
}

template <class T>
[[nodiscard]] consteval Type type() {
  using Value = std::remove_cv_t<T>;
  if constexpr (std::is_same_v<Value, std::int32_t>) {
    return Type::I32;
  } else if constexpr (std::is_same_v<Value, std::uint32_t>) {
    return Type::U32;
  } else if constexpr (std::is_same_v<Value, std::int64_t>) {
    return Type::I64;
  } else if constexpr (std::is_same_v<Value, std::uint64_t>) {
    return Type::U64;
  } else if constexpr (FixedValue<Value> && sizeof(Value) == 4u) {
    return Type::FixedLane32;
  } else if constexpr (FixedValue<Value> && sizeof(Value) == 8u) {
    return Type::FixedLane64;
  } else {
    static_assert(UnsupportedType<Value>,
                  "runD Compute supports integer and fixed domains only");
  }
}

} // namespace rund::compute::detail

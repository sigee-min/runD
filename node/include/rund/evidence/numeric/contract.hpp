#pragma once

#include <cstdint>

namespace rund::evidence {

enum class Domain : std::uint8_t {
  I32,
  U32,
  I64,
  U64,
  Fixed,
  F32,
  F64,
};

enum class Rounding : std::uint8_t {
  TowardZero,
  Down,
  Up,
  NearestEven,
};

enum class Overflow : std::uint8_t {
  Saturate,
  Wrap,
};

enum class Approximation : std::uint8_t {
  Exact,
  Deterministic,
};

enum class Arithmetic : std::uint8_t {
  IntegerWrap,
  IntegerSaturating,
  FixedPoint,
  FloatingPoint,
  StrictFloatingPoint,
  ReductionSlot,
  HashDigest,
};

enum class Authority : std::uint8_t {
  Authoritative,
  Derived,
  PresentationOnly,
  Diagnostic,
};

enum class Determinism : std::uint8_t {
  Required,
  BestEffort,
  NotRequired,
};

struct Id {
  std::uint64_t value = 0u;
  [[nodiscard]] constexpr explicit operator bool() const { return value != 0u; }
  friend constexpr bool operator==(const Id &, const Id &) = default;
};

struct Contract {
  Domain domain = Domain::I32;
  Arithmetic arithmetic = Arithmetic::IntegerWrap;
  Authority authority = Authority::Authoritative;
  Determinism determinism = Determinism::Required;
  std::uint8_t integer_bits = 0u;
  std::uint8_t fraction_bits = 0u;
  Rounding rounding = Rounding::NearestEven;
  Overflow overflow = Overflow::Saturate;
  Approximation approximation = Approximation::Exact;

  friend constexpr bool operator==(const Contract &,
                                   const Contract &) = default;
};

namespace detail::contract {

inline constexpr std::uint32_t schema_tag = 1u;

[[nodiscard]] constexpr bool is_integer_domain(const Domain domain) {
  return domain == Domain::I32 || domain == Domain::U32 ||
         domain == Domain::I64 || domain == Domain::U64;
}

[[nodiscard]] constexpr bool is_fixed_domain(const Domain domain) {
  return domain == Domain::Fixed;
}

[[nodiscard]] constexpr bool is_floating_domain(const Domain domain) {
  return domain == Domain::F32 || domain == Domain::F64;
}

[[nodiscard]] constexpr bool valid_domain(const Domain domain) {
  switch (domain) {
  case Domain::I32:
  case Domain::U32:
  case Domain::I64:
  case Domain::U64:
  case Domain::Fixed:
  case Domain::F32:
  case Domain::F64:
    return true;
  }
  return false;
}

[[nodiscard]] constexpr bool valid_fixed_policy(const Contract contract) {
  const std::uint16_t width =
      static_cast<std::uint16_t>(contract.integer_bits) +
      contract.fraction_bits;
  return contract.integer_bits >= 1u && contract.fraction_bits >= 1u &&
         (width == 32u || width == 64u) &&
         static_cast<std::uint8_t>(contract.rounding) <=
             static_cast<std::uint8_t>(Rounding::NearestEven) &&
         static_cast<std::uint8_t>(contract.overflow) <=
             static_cast<std::uint8_t>(Overflow::Wrap) &&
         static_cast<std::uint8_t>(contract.approximation) <=
             static_cast<std::uint8_t>(Approximation::Deterministic);
}

[[nodiscard]] constexpr bool
canonical_non_fixed_policy(const Contract contract) {
  return contract.integer_bits == 0u && contract.fraction_bits == 0u &&
         contract.rounding == Rounding::NearestEven &&
         contract.overflow == Overflow::Saturate &&
         contract.approximation == Approximation::Exact;
}

[[nodiscard]] constexpr bool valid_arithmetic(const Arithmetic arithmetic) {
  switch (arithmetic) {
  case Arithmetic::IntegerWrap:
  case Arithmetic::IntegerSaturating:
  case Arithmetic::FixedPoint:
  case Arithmetic::FloatingPoint:
  case Arithmetic::StrictFloatingPoint:
  case Arithmetic::ReductionSlot:
  case Arithmetic::HashDigest:
    return true;
  }
  return false;
}

[[nodiscard]] constexpr bool valid_authority(const Authority authority) {
  switch (authority) {
  case Authority::Authoritative:
  case Authority::Derived:
  case Authority::PresentationOnly:
  case Authority::Diagnostic:
    return true;
  }
  return false;
}

[[nodiscard]] constexpr bool valid_determinism(const Determinism determinism) {
  switch (determinism) {
  case Determinism::Required:
  case Determinism::BestEffort:
  case Determinism::NotRequired:
    return true;
  }
  return false;
}

} // namespace detail::contract

[[nodiscard]] constexpr bool valid(const Contract contract) noexcept {
  if (!detail::contract::valid_domain(contract.domain) ||
      !detail::contract::valid_arithmetic(contract.arithmetic) ||
      !detail::contract::valid_authority(contract.authority) ||
      !detail::contract::valid_determinism(contract.determinism)) {
    return false;
  }
  if (contract.authority == Authority::Authoritative &&
      contract.determinism != Determinism::Required) {
    return false;
  }
  if (detail::contract::is_fixed_domain(contract.domain)
          ? !detail::contract::valid_fixed_policy(contract)
          : !detail::contract::canonical_non_fixed_policy(contract)) {
    return false;
  }
  switch (contract.arithmetic) {
  case Arithmetic::IntegerWrap:
  case Arithmetic::IntegerSaturating:
    return detail::contract::is_integer_domain(contract.domain);
  case Arithmetic::FixedPoint:
    return detail::contract::is_fixed_domain(contract.domain);
  case Arithmetic::FloatingPoint:
    return detail::contract::is_floating_domain(contract.domain) &&
           contract.authority != Authority::Authoritative;
  case Arithmetic::StrictFloatingPoint:
    return detail::contract::is_floating_domain(contract.domain);
  case Arithmetic::ReductionSlot:
  case Arithmetic::HashDigest:
    return contract.domain == Domain::U64;
  }
  return false;
}

[[nodiscard]] constexpr Id identify(const Contract contract) noexcept {
  if (!valid(contract)) {
    return {};
  }
  constexpr std::uint64_t offset = 14695981039346656037ull;
  constexpr std::uint64_t prime = 1099511628211ull;
  const auto mix = [](const std::uint64_t hash,
                      const std::uint64_t value) constexpr {
    return (hash ^ value) * prime;
  };
  std::uint64_t hash = offset;
  hash = mix(hash, detail::contract::schema_tag);
  hash = mix(hash, static_cast<std::uint8_t>(contract.domain));
  hash = mix(hash, static_cast<std::uint8_t>(contract.arithmetic));
  hash = mix(hash, static_cast<std::uint8_t>(contract.authority));
  hash = mix(hash, static_cast<std::uint8_t>(contract.determinism));
  hash = mix(hash, contract.integer_bits);
  hash = mix(hash, contract.fraction_bits);
  hash = mix(hash, static_cast<std::uint8_t>(contract.rounding));
  hash = mix(hash, static_cast<std::uint8_t>(contract.overflow));
  hash = mix(hash, static_cast<std::uint8_t>(contract.approximation));
  return Id{hash == 0u ? 1u : hash};
}

} // namespace rund::evidence

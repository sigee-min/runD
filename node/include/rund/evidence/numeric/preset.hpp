#pragma once

#include <rund/evidence/numeric/contract.hpp>

namespace rund::evidence {

[[nodiscard]] constexpr Contract i32() {
  return Contract{
      .domain = Domain::I32,
      .arithmetic = Arithmetic::IntegerWrap,
      .authority = Authority::Authoritative,
      .determinism = Determinism::Required,
  };
}

[[nodiscard]] constexpr Contract i64() {
  return Contract{
      .domain = Domain::I64,
      .arithmetic = Arithmetic::IntegerWrap,
      .authority = Authority::Authoritative,
      .determinism = Determinism::Required,
  };
}

template <std::uint8_t IntegerBits, std::uint8_t FractionBits>
  requires(IntegerBits >= 1u && FractionBits >= 1u &&
           (static_cast<unsigned>(IntegerBits) + FractionBits == 32u ||
            static_cast<unsigned>(IntegerBits) + FractionBits == 64u))
[[nodiscard]] constexpr Contract fixed() {
  return Contract{
      .domain = Domain::Fixed,
      .arithmetic = Arithmetic::FixedPoint,
      .authority = Authority::Authoritative,
      .determinism = Determinism::Required,
      .integer_bits = IntegerBits,
      .fraction_bits = FractionBits,
  };
}

[[nodiscard]] constexpr Contract strict_f32() {
  return Contract{
      .domain = Domain::F32,
      .arithmetic = Arithmetic::StrictFloatingPoint,
      .authority = Authority::Authoritative,
      .determinism = Determinism::Required,
  };
}

[[nodiscard]] constexpr Contract strict_f64() {
  return Contract{
      .domain = Domain::F64,
      .arithmetic = Arithmetic::StrictFloatingPoint,
      .authority = Authority::Authoritative,
      .determinism = Determinism::Required,
  };
}

[[nodiscard]] constexpr Contract diagnostic_f32() {
  return Contract{
      .domain = Domain::F32,
      .arithmetic = Arithmetic::FloatingPoint,
      .authority = Authority::Diagnostic,
      .determinism = Determinism::BestEffort,
  };
}

[[nodiscard]] constexpr Contract diagnostic_f64() {
  return Contract{
      .domain = Domain::F64,
      .arithmetic = Arithmetic::FloatingPoint,
      .authority = Authority::Diagnostic,
      .determinism = Determinism::BestEffort,
  };
}

[[nodiscard]] constexpr Contract presentation_f32() {
  return Contract{
      .domain = Domain::F32,
      .arithmetic = Arithmetic::FloatingPoint,
      .authority = Authority::PresentationOnly,
      .determinism = Determinism::NotRequired,
  };
}

[[nodiscard]] constexpr Contract presentation_f64() {
  return Contract{
      .domain = Domain::F64,
      .arithmetic = Arithmetic::FloatingPoint,
      .authority = Authority::PresentationOnly,
      .determinism = Determinism::NotRequired,
  };
}

} // namespace rund::evidence

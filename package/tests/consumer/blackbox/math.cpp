#include "model.hpp"

#include <math32/math32.hpp>
#include <math64/math64.hpp>

namespace package_blackbox {

[[nodiscard]] int CheckMathAndEvidence() {
  static_assert(sizeof(rund::math32::i32) == 4u);
  static_assert(sizeof(rund::math64::i64) == 8u);
  static_assert(rund::math32::FixedScale == (rund::math32::u64{1} << 31u));
  static_assert(rund::math64::FixedScale == (rund::math64::u64{1} << 63u));

  if (!rund::math32::simd::Any(rund::math32::simd::MaskTrue()) ||
      !rund::math64::simd::Any(rund::math64::simd::MaskTrue())) {
    return Mismatch("math-simd");
  }

  const rund::evidence::Numeric evidence =
      rund::evidence::make(rund::evidence::Contract{
          .domain = rund::evidence::Domain::F32,
          .arithmetic = rund::evidence::Arithmetic::StrictFloatingPoint,
          .authority = rund::evidence::Authority::Authoritative,
          .determinism = rund::evidence::Determinism::Required,
      });
  if (!evidence) {
    return evidence.exit_code();
  }
  if (evidence.hash() == 0u || !evidence.strict_float()) {
    return Mismatch("numeric-evidence");
  }

  const rund::evidence::Numeric decoded =
      rund::evidence::decode(rund::evidence::encode(evidence));
  if (!decoded) {
    return decoded.exit_code();
  }
  return decoded.hash() == evidence.hash() && decoded.id() == evidence.id()
             ? 0
             : Mismatch("numeric-evidence-codec");
}

} // namespace package_blackbox

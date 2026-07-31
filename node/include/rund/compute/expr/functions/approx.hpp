#pragma once

#include <rund/compute/expr/functions/algebra.hpp>

namespace rund::compute {

struct ActivationOp final {
  struct ReluTag final {};
  struct LeakyReluTag final {};
  struct HardSigmoidTag final {};
  struct HardSwishTag final {};
  struct HardTanhTag final {};

  inline static constexpr ReluTag Relu{};
  inline static constexpr LeakyReluTag LeakyRelu{};
  inline static constexpr HardSigmoidTag HardSigmoid{};
  inline static constexpr HardSwishTag HardSwish{};
  inline static constexpr HardTanhTag HardTanh{};
};
struct WindowOp final {
  struct ParabolicTag final {};
  struct TriangularTag final {};
  struct HannTag final {};
  struct HammingTag final {};
  struct BlackmanTag final {};
  struct LanczosTag final {};

  inline static constexpr ParabolicTag Parabolic{};
  inline static constexpr TriangularTag Triangular{};
  inline static constexpr HannTag Hann{};
  inline static constexpr HammingTag Hamming{};
  inline static constexpr BlackmanTag Blackman{};
  inline static constexpr LanczosTag Lanczos{};
};
struct ComplexOp final {
  struct ConjTag final {};
  struct Abs2Tag final {};
  struct AbsTag final {};
  struct PhaseTag final {};
  struct MulTag final {};

  inline static constexpr ConjTag Conj{};
  inline static constexpr Abs2Tag Abs2{};
  inline static constexpr AbsTag Abs{};
  inline static constexpr PhaseTag Phase{};
  inline static constexpr MulTag Mul{};
};
struct ComplexPart final {
  struct RealTag final {};
  struct ImagTag final {};

  inline static constexpr RealTag Real{};
  inline static constexpr ImagTag Imag{};
};

namespace detail {
template <class E>
  requires FixedExpression<E>
[[nodiscard]] constexpr auto q31_like(const E &anchor,
                                      const std::uint32_t bits) {
  return ratio_like(anchor, static_cast<std::int64_t>(bits),
                    std::int64_t{1} << 31u);
}
} // namespace detail

[[nodiscard]] constexpr auto activation(ActivationOp::ReluTag,
                                        const auto &value) {
  return positive_part(value);
}
[[nodiscard]] constexpr auto activation(ActivationOp::ReluTag,
                                        const auto &value, const auto &upper) {
  return min(positive_part(value), positive_part(upper));
}
[[nodiscard]] constexpr auto activation(ActivationOp::LeakyReluTag,
                                        const auto &value, const auto &slope) {
  return select(is_nonneg(value), value, mul_fixed(value, slope));
}
[[nodiscard]] constexpr auto activation(ActivationOp::HardSigmoidTag,
                                        const auto &value) {
  return saturate(mean(value, fixed_one(value)));
}
[[nodiscard]] constexpr auto activation(ActivationOp::HardSwishTag,
                                        const auto &value) {
  return mul_fixed(value, activation(ActivationOp::HardSigmoid, value));
}
[[nodiscard]] constexpr auto activation(ActivationOp::HardTanhTag,
                                        const auto &value) {
  return clip(value, fixed_one(value));
}

[[nodiscard]] constexpr auto softsign(const auto &value, const auto &scale) {
  const auto half = fixed(FixedOp::Half, value);
  const auto positive_scale = max(detail::storage(abs(scale)), half);
  return ratio(value, add_sat(positive_scale,
                              mul_fixed(detail::storage(abs(value)), half)));
}
[[nodiscard]] constexpr auto softsign(const auto &value) {
  return softsign(value, fixed(FixedOp::Half, value));
}

[[nodiscard]] constexpr auto huber(const auto &value, const auto &delta) {
  const auto bound = detail::storage(abs(delta));
  const auto magnitude = detail::storage(abs(value));
  const auto half = fixed(FixedOp::Half, value);
  const auto quadratic = mul_fixed(half, mul_fixed(value, value));
  const auto linear =
      mul_fixed(bound, sub_sat(magnitude, mul_fixed(half, bound)));
  return select(magnitude <= bound, quadratic, linear);
}

[[nodiscard]] constexpr auto smootherstep(const auto &edge0, const auto &edge1,
                                          const auto &value) {
  const auto t = unlerp(edge0, edge1, value);
  const auto inverse = sub_sat(fixed_one(t), t);
  const auto t2 = mul_fixed(t, t);
  const auto t3 = mul_fixed(t2, t);
  const auto t4 = mul_fixed(t3, t);
  const auto t5 = mul_fixed(t4, t);
  const auto inverse2 = mul_fixed(inverse, inverse);
  const auto first = mul_fixed(t3, inverse2);
  const auto second = mul_fixed(t4, inverse);
  const auto five_second = add_sat(add_sat(second, second),
                                   add_sat(add_sat(second, second), second));
  const auto five_first =
      add_sat(add_sat(first, first), add_sat(add_sat(first, first), first));
  return add_sat(add_sat(add_sat(five_first, five_first), five_second), t5);
}

[[nodiscard]] constexpr auto window(WindowOp::ParabolicTag,
                                    const auto &amount) {
  const auto t = saturate(amount);
  const auto q = mul_fixed(t, sub_sat(fixed_one(t), t));
  return add_sat(add_sat(q, q), add_sat(q, q));
}
[[nodiscard]] constexpr auto window(WindowOp::TriangularTag,
                                    const auto &amount) {
  const auto t = saturate(amount);
  const auto delta =
      detail::storage(centered(CenteredOp::Abs, t, fixed(FixedOp::Half, t)));
  return sub_sat(fixed_one(t), add_sat(delta, delta));
}
[[nodiscard]] constexpr auto window(WindowOp::HannTag, const auto &amount) {
  const auto t = saturate(amount);
  const auto half = fixed(FixedOp::Half, t);
  return sub_sat(half, mul_fixed(half, cos(t)));
}
[[nodiscard]] constexpr auto window(WindowOp::HammingTag, const auto &amount) {
  const auto t = saturate(amount);
  const auto a = detail::q31_like(t, 0x451eb852u);
  const auto b = detail::q31_like(t, 0x3ae147aeu);
  return sub_sat(a, mul_fixed(b, cos(t)));
}
[[nodiscard]] constexpr auto window(WindowOp::BlackmanTag, const auto &amount) {
  const auto t = saturate(amount);
  const auto a0 = detail::q31_like(t, 0x35c28f5cu);
  const auto a1 = fixed(FixedOp::Half, t);
  const auto a2 = detail::q31_like(t, 0x0a3d70a3u);
  return add_sat(sub_sat(a0, mul_fixed(a1, cos(t))),
                 mul_fixed(a2, cos(add_sat(t, t))));
}
[[nodiscard]] constexpr auto window(WindowOp::LanczosTag, const auto &amount) {
  const auto t = saturate(amount);
  const auto centered_value = sub_sat(t, fixed(FixedOp::Half, t));
  return div_fixed(
      sin(centered_value),
      max(detail::storage(abs(centered_value)), fixed(FixedOp::Quarter, t)));
}

[[nodiscard]] constexpr auto complex(ComplexOp::ConjTag, ComplexPart::RealTag,
                                     const auto &real, const auto &) {
  return real;
}
[[nodiscard]] constexpr auto complex(ComplexOp::ConjTag, ComplexPart::ImagTag,
                                     const auto &, const auto &imag) {
  return -imag;
}
[[nodiscard]] constexpr auto complex(ComplexOp::Abs2Tag, const auto &real,
                                     const auto &imag) {
  return add_sat(mul_fixed(real, real), mul_fixed(imag, imag));
}
[[nodiscard]] constexpr auto complex(ComplexOp::AbsTag, const auto &real,
                                     const auto &imag) {
  return sqrt(complex(ComplexOp::Abs2, real, imag));
}
[[nodiscard]] constexpr auto complex(ComplexOp::PhaseTag, const auto &real,
                                     const auto &imag) {
  return atan2(imag, real);
}
[[nodiscard]] constexpr auto complex(ComplexOp::MulTag, ComplexPart::RealTag,
                                     const auto &ar, const auto &ai,
                                     const auto &br, const auto &bi) {
  return sub_sat(mul_fixed(ar, br), mul_fixed(ai, bi));
}
[[nodiscard]] constexpr auto complex(ComplexOp::MulTag, ComplexPart::ImagTag,
                                     const auto &ar, const auto &ai,
                                     const auto &br, const auto &bi) {
  return add_sat(mul_fixed(ar, bi), mul_fixed(ai, br));
}

} // namespace rund::compute

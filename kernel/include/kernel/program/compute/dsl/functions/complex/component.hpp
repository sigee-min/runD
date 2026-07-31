#pragma once

namespace rund::compute_dsl {

struct ComplexPartReal final {};
struct ComplexPartImag final {};

struct ComplexPart final {
  inline static constexpr ComplexPartReal Real{};
  inline static constexpr ComplexPartImag Imag{};
};

struct ComplexOpConj final {};
struct ComplexOpMul final {};
struct ComplexOpAbs final {};
struct ComplexOpAbs2 final {};
struct ComplexOpPhase final {};

struct ComplexOp final {
  inline static constexpr ComplexOpConj Conj{};
  inline static constexpr ComplexOpMul Mul{};
  inline static constexpr ComplexOpAbs Abs{};
  inline static constexpr ComplexOpAbs2 Abs2{};
  inline static constexpr ComplexOpPhase Phase{};
};

[[nodiscard]] inline ComputeValue complex(const ComplexOpConj,
                                          const ComplexPartReal,
                                          const ComputeValue real,
                                          const ComputeValue) noexcept {
  return real;
}

[[nodiscard]] inline ComputeValue complex(const ComplexOpConj,
                                          const ComplexPartImag,
                                          const ComputeValue,
                                          const ComputeValue imag) noexcept {
  return detail::Unary(rund::kernel::IrOp::Neg, imag);
}

} // namespace rund::compute_dsl

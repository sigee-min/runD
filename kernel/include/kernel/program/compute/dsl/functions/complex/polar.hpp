#pragma once

namespace rund::compute_dsl {

[[nodiscard]] inline ComputeValue complex(const ComplexOpAbs2,
                                          const ComputeValue real,
                                          const ComputeValue imag) noexcept {
  return add_sat(mul_fixed(real, real), mul_fixed(imag, imag));
}

[[nodiscard]] inline ComputeValue complex(const ComplexOpAbs op,
                                          const ComputeValue real,
                                          const ComputeValue imag) noexcept {
  (void)op;
  return sqrt(complex(ComplexOp::Abs2, real, imag));
}

[[nodiscard]] inline ComputeValue complex(const ComplexOpPhase,
                                          const ComputeValue real,
                                          const ComputeValue imag) noexcept {
  return detail::Binary(rund::kernel::IrOp::Atan2, imag, real);
}

} // namespace rund::compute_dsl

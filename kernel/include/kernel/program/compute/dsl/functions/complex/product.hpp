#pragma once

namespace rund::compute_dsl {

[[nodiscard]] inline ComputeValue complex(const ComplexOpMul,
                                          const ComplexPartReal,
                                          const ComputeValue ar,
                                          const ComputeValue ai,
                                          const ComputeValue br,
                                          const ComputeValue bi) noexcept {
  return sub_sat(mul_fixed(ar, br), mul_fixed(ai, bi));
}

[[nodiscard]] inline ComputeValue complex(const ComplexOpMul,
                                          const ComplexPartImag,
                                          const ComputeValue ar,
                                          const ComputeValue ai,
                                          const ComputeValue br,
                                          const ComputeValue bi) noexcept {
  return add_sat(mul_fixed(ar, bi), mul_fixed(ai, br));
}

} // namespace rund::compute_dsl

#include "shape.hpp"

namespace rund::node::accel::detail {

bool TransformShapeOk(const rund::kernel::TransformDesc &desc,
                      const rund::kernel::TransformPlan &plan,
                      const TransformBinds &bindings) noexcept {
  if (!TransformDescMatchesPlan(desc, plan) ||
      bindings.input_real_handle == nullptr ||
      bindings.input_imag_handle == nullptr ||
      bindings.output_real_handle == nullptr ||
      bindings.output_imag_handle == nullptr ||
      bindings.input_real == nullptr) {
    return false;
  }
  if (bindings.input_imag == nullptr || bindings.output_real == nullptr ||
      bindings.output_imag == nullptr) {
    return false;
  }
  const std::uint64_t bytes = bindings.input_real->element_bytes;
  if (bytes != plan.element_bytes ||
      !PrimitiveResidentExactShapeOk(bindings.input_real, bytes,
                                     plan.element_count,
                                     rund::kernel::kResidentUsageRead) ||
      !PrimitiveResidentExactShapeOk(bindings.input_imag, bytes,
                                     plan.element_count,
                                     rund::kernel::kResidentUsageRead) ||
      !PrimitiveResidentExactShapeOk(bindings.output_real, bytes,
                                     plan.element_count,
                                     rund::kernel::kResidentUsageWrite) ||
      !PrimitiveResidentExactShapeOk(bindings.output_imag, bytes,
                                     plan.element_count,
                                     rund::kernel::kResidentUsageWrite)) {
    return false;
  }
  return !ResidentOverlap(*bindings.output_real, *bindings.output_imag) &&
         !ResidentOverlap(*bindings.output_real, *bindings.input_real) &&
         !ResidentOverlap(*bindings.output_real, *bindings.input_imag) &&
         !ResidentOverlap(*bindings.output_imag, *bindings.input_real) &&
         !ResidentOverlap(*bindings.output_imag, *bindings.input_imag);
}

} // namespace rund::node::accel::detail

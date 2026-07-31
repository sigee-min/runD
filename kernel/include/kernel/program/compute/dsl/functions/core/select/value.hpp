#pragma once

namespace rund::compute_dsl::detail {
[[nodiscard]] inline ComputeValue SelectValue(const ComputeValue condition,
                                              const ComputeValue when_true,
                                              const ComputeValue when_false) noexcept {
  return Ternary(rund::kernel::IrOp::Select, condition, when_true, when_false);
}
template <ConstantLiteral T>
[[nodiscard]] inline ComputeValue SelectFalse(const ComputeValue condition,
                                              const ComputeValue when_true,
                                              const T when_false) noexcept {
  return SelectValue(condition, when_true, ConstantValue(when_true, when_false));
}
template <ConstantLiteral T>
[[nodiscard]] inline ComputeValue SelectTrue(const ComputeValue condition,
                                             const T when_true,
                                             const ComputeValue when_false) noexcept {
  return SelectValue(condition, ConstantValue(when_false, when_true), when_false);
}
template <ConstantLiteral T, ConstantLiteral U>
[[nodiscard]] inline ComputeValue SelectBranches(const ComputeValue condition,
                                                 const T when_true,
                                                 const U when_false) noexcept {
  return SelectValue(condition, StorageConstantValue(condition, when_true),
                     StorageConstantValue(condition, when_false));
}
} // namespace rund::compute_dsl::detail

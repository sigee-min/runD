#pragma once

#include <kernel/program/compute/dsl/expression/context.hpp>

#include <memory>

namespace rund::compute_dsl::detail {

class Expr {
public:
  constexpr Expr() noexcept = default;

private:
  struct InternalToken {};

  Expr(InternalToken, BuildContext *context, rund::kernel::u32 node) noexcept;

  [[nodiscard]] BuildContext *context() const noexcept;
  [[nodiscard]] rund::kernel::u32 node() const noexcept;

  BuildContext *context_ = nullptr;
  std::weak_ptr<BuildContext::ContextLifetime> lifetime_;
  rund::kernel::u32 node_ = 0u;

  friend Expr Binary(rund::kernel::IrOp op, Expr lhs, Expr rhs) noexcept;
  friend Expr Unary(rund::kernel::IrOp op, Expr value) noexcept;
  friend Expr ConstShift(rund::kernel::IrOp op, Expr value,
                         rund::kernel::u32 amount) noexcept;
  friend Expr Ternary(rund::kernel::IrOp op, Expr lhs, Expr rhs,
                      Expr aux) noexcept;
  friend Expr Constant(Expr anchor, rund::kernel::u64 bits) noexcept;
  friend Expr TypedConstant(Expr anchor, ScalarMode mode,
                            rund::kernel::u64 bits) noexcept;
  friend Expr
  FormattedConstant(Expr anchor, rund::kernel::u64 bits,
                    rund::kernel::ComputeFixedFormat format) noexcept;
  friend Expr StorageConstant(Expr anchor, rund::kernel::u64 bits) noexcept;
  friend Expr StorageQuantize(Expr value) noexcept;
  friend Expr UnaryFormatted(rund::kernel::IrOp op, Expr value,
                             rund::kernel::ComputeFixedFormat format) noexcept;
  friend Expr BinaryFormatted(rund::kernel::IrOp op, Expr lhs, Expr rhs,
                              rund::kernel::ComputeFixedFormat format) noexcept;
  friend Expr
  TernaryFormatted(rund::kernel::IrOp op, Expr first, Expr second, Expr third,
                   rund::kernel::ComputeFixedFormat format) noexcept;
  friend Expr DynamicRead(BuildContext &context,
                          rund::kernel::u32 binding) noexcept;
  friend Expr DynamicRead(BuildContext &context, rund::kernel::u32 binding,
                          rund::kernel::ComputeFixedFormat format) noexcept;
  friend Expr DynamicUniformRead(
      BuildContext &context, rund::kernel::u32 binding,
      rund::kernel::ComputeFixedFormat format) noexcept;
  friend Expr DynamicReadAt(BuildContext &context,
                            rund::kernel::u32 binding,
                            rund::kernel::u32 index,
                            rund::kernel::u32 count,
                            rund::kernel::ComputeFixedFormat format) noexcept;
  friend Expr DynamicIndex(BuildContext &context) noexcept;
  friend Expr DynamicIndex(BuildContext &context, ScalarMode mode) noexcept;
  friend Expr TypedIndex(Expr anchor, ScalarMode mode) noexcept;
  friend void DynamicWrite(BuildContext &context, rund::kernel::u32 binding,
                           Expr value) noexcept;
  friend void DynamicCheckedOrdinalWrite(BuildContext &context,
                                         rund::kernel::u32 binding,
                                         Expr value) noexcept;
  friend void DynamicBoundaryMaskWrite(
      BuildContext &context, rund::kernel::u32 binding, Expr value,
      rund::kernel::ComputeFixedFormat target_format) noexcept;
  friend ScalarMode ScalarModeOf(Expr value) noexcept;
  friend rund::kernel::ComputeFixedFormat FixedFormatOf(Expr value) noexcept;
  friend class ReadHandle;
  friend class WriteTarget;
  template <typename Body> friend class Access;
};

[[nodiscard]] Expr DynamicRead(BuildContext &context,
                               rund::kernel::u32 binding) noexcept;
[[nodiscard]] Expr
DynamicRead(BuildContext &context, rund::kernel::u32 binding,
            rund::kernel::ComputeFixedFormat format) noexcept;
[[nodiscard]] Expr DynamicUniformRead(
    BuildContext &context, rund::kernel::u32 binding,
    rund::kernel::ComputeFixedFormat format = {}) noexcept;
[[nodiscard]] Expr DynamicReadAt(
    BuildContext &context, rund::kernel::u32 binding,
    rund::kernel::u32 index, rund::kernel::u32 count,
    rund::kernel::ComputeFixedFormat format = {}) noexcept;
[[nodiscard]] Expr DynamicIndex(BuildContext &context) noexcept;
[[nodiscard]] Expr DynamicIndex(BuildContext &context,
                                ScalarMode mode) noexcept;
[[nodiscard]] Expr TypedIndex(Expr anchor, ScalarMode mode) noexcept;
void DynamicWrite(BuildContext &context, rund::kernel::u32 binding,
                  Expr value) noexcept;
void DynamicCheckedOrdinalWrite(BuildContext &context,
                                rund::kernel::u32 binding, Expr value) noexcept;
void DynamicBoundaryMaskWrite(
    BuildContext &context, rund::kernel::u32 binding, Expr value,
    rund::kernel::ComputeFixedFormat target_format) noexcept;

[[nodiscard]] Expr Binary(rund::kernel::IrOp op, Expr lhs, Expr rhs) noexcept;
[[nodiscard]] Expr Unary(rund::kernel::IrOp op, Expr value) noexcept;
[[nodiscard]] Expr ConstShift(rund::kernel::IrOp op, Expr value,
                              rund::kernel::u32 amount) noexcept;
[[nodiscard]] Expr Ternary(rund::kernel::IrOp op, Expr lhs, Expr rhs,
                           Expr aux) noexcept;
[[nodiscard]] Expr Constant(Expr anchor, rund::kernel::u64 bits) noexcept;
[[nodiscard]] Expr TypedConstant(Expr anchor, ScalarMode mode,
                                 rund::kernel::u64 bits) noexcept;
[[nodiscard]] Expr
FormattedConstant(Expr anchor, rund::kernel::u64 bits,
                  rund::kernel::ComputeFixedFormat format) noexcept;
[[nodiscard]] Expr StorageConstant(Expr anchor,
                                   rund::kernel::u64 bits) noexcept;
[[nodiscard]] Expr StorageQuantize(Expr value) noexcept;
[[nodiscard]] Expr
UnaryFormatted(rund::kernel::IrOp op, Expr value,
               rund::kernel::ComputeFixedFormat format) noexcept;
[[nodiscard]] Expr
BinaryFormatted(rund::kernel::IrOp op, Expr lhs, Expr rhs,
                rund::kernel::ComputeFixedFormat format) noexcept;
[[nodiscard]] Expr
TernaryFormatted(rund::kernel::IrOp op, Expr first, Expr second, Expr third,
                 rund::kernel::ComputeFixedFormat format) noexcept;
[[nodiscard]] ScalarMode ScalarModeOf(Expr value) noexcept;
[[nodiscard]] rund::kernel::ComputeFixedFormat
FixedFormatOf(Expr value) noexcept;

} // namespace rund::compute_dsl::detail

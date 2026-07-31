#pragma once

[[nodiscard]] constexpr bool UnaryValueOp(const IrOp op) noexcept {
  return op == IrOp::Neg || op == IrOp::Abs || op == IrOp::AbsMagnitude ||
         op == IrOp::Sign || op == IrOp::Quantize || op == IrOp::PredicateNot ||
         op == IrOp::BitNot || op == IrOp::NegPositiveFixed ||
         op == IrOp::Recip || op == IrOp::Sqrt || op == IrOp::Rsqrt ||
         op == IrOp::Sin || op == IrOp::Cos || op == IrOp::Tan ||
         op == IrOp::Exp || op == IrOp::Log;
}
[[nodiscard]] constexpr bool BinaryValueOp(const IrOp op) noexcept {
  return op == IrOp::Add || op == IrOp::Sub || op == IrOp::Mul ||
         op == IrOp::MulWrap || op == IrOp::Min || op == IrOp::Max ||
         op == IrOp::Eq || op == IrOp::Lt || op == IrOp::Le || op == IrOp::Ne ||
         op == IrOp::Gt || op == IrOp::Ge || op == IrOp::PredicateAnd ||
         op == IrOp::PredicateOr || op == IrOp::BitAnd || op == IrOp::BitOr ||
         op == IrOp::BitXor || op == IrOp::AddSat ||
         op == IrOp::AddSatUnsigned || op == IrOp::SubSat ||
         op == IrOp::MulFixed || op == IrOp::MulFixedScaled ||
         op == IrOp::MulUnsignedFixed || op == IrOp::DivFixed ||
         op == IrOp::Atan2 || op == IrOp::DivSigned ||
         op == IrOp::DivUnsigned || op == IrOp::MinUnsigned ||
         op == IrOp::MaxUnsigned || op == IrOp::LtUnsigned ||
         op == IrOp::LeUnsigned || op == IrOp::GtUnsigned ||
         op == IrOp::GeUnsigned;
}

[[nodiscard]] constexpr bool ConstShiftOp(const IrOp op) noexcept {
  return op == IrOp::ShlConst || op == IrOp::ShrLogicalConst ||
         op == IrOp::ShrArithmeticConst;
}

[[nodiscard]] constexpr bool TernaryValueOp(const IrOp op) noexcept {
  return op == IrOp::Clamp || op == IrOp::Select || op == IrOp::MulAddFixed ||
         op == IrOp::ClampUnsigned;
}

[[nodiscard]] constexpr u32
ScalarBitWidth(const ComputeScalar scalar) noexcept {
  return scalar == ComputeScalar::Lane64 ? 64u : 32u;
}

[[nodiscard]] constexpr u32 ScalarBytes(const ComputeScalar scalar) noexcept {
  return scalar == ComputeScalar::Lane64 ? 8u : 4u;
}

#pragma once

#include <kernel/program/compute/lowering/model.hpp>

namespace rund::kernel {
namespace compute_lowering_detail {

[[nodiscard]] constexpr bool
UnsignedDomain(const ComputeDomain domain) noexcept {
  return domain == ComputeDomain::U32 || domain == ComputeDomain::U64;
}

[[nodiscard]] constexpr u8 ScalarModeFor(const ComputeScalar scalar) noexcept {
  return scalar == ComputeScalar::Lane64 ? u8{2u} : u8{1u};
}

[[nodiscard]] constexpr u8 DomainModeFor(const ComputeScalar scalar,
                                         const ComputeDomain domain) noexcept {
  switch (domain) {
  case ComputeDomain::I32:
    return 3u;
  case ComputeDomain::U32:
    return 4u;
  case ComputeDomain::I64:
    return 5u;
  case ComputeDomain::U64:
    return 6u;
  case ComputeDomain::Fixed:
    return ScalarModeFor(scalar);
  }
  return 0u;
}

[[nodiscard]] constexpr ComputeDomain DomainForMode(const ComputeScalar scalar,
                                                    const u8 mode) noexcept {
  if (mode == ScalarModeFor(scalar)) {
    return ComputeDomain::Fixed;
  }
  switch (mode) {
  case 3u:
    return scalar == ComputeScalar::Lane32 ? ComputeDomain::I32
                                           : static_cast<ComputeDomain>(0u);
  case 4u:
    return scalar == ComputeScalar::Lane32 ? ComputeDomain::U32
                                           : static_cast<ComputeDomain>(0u);
  case 5u:
    return scalar == ComputeScalar::Lane64 ? ComputeDomain::I64
                                           : static_cast<ComputeDomain>(0u);
  case 6u:
    return scalar == ComputeScalar::Lane64 ? ComputeDomain::U64
                                           : static_cast<ComputeDomain>(0u);
  default:
    return static_cast<ComputeDomain>(0u);
  }
}

[[nodiscard]] constexpr const char *
DomainName(const ComputeScalar scalar, const ComputeDomain domain) noexcept {
  switch (domain) {
  case ComputeDomain::I32:
    return "i32";
  case ComputeDomain::U32:
    return "u32";
  case ComputeDomain::I64:
    return "i64";
  case ComputeDomain::U64:
    return "u64";
  case ComputeDomain::Fixed:
    return scalar == ComputeScalar::Lane64 ? "fixed_lane64" : "fixed_lane32";
  }
  return "unknown";
}

[[nodiscard]] constexpr const char *BindingKindName(const u8 kind) noexcept {
  switch (kind) {
  case 1u:
    return "param";
  case 2u:
    return "read";
  case 3u:
    return "write";
  default:
    return nullptr;
  }
}

[[nodiscard]] constexpr const char *
ScalarName(const ComputeScalar scalar) noexcept {
  switch (scalar) {
  case ComputeScalar::Lane32:
    return "fixed_lane32";
  case ComputeScalar::Lane64:
    return "fixed_lane64";
  }
  return "unknown";
}

[[nodiscard]] constexpr const char *ApiName(const ComputeApi api) noexcept {
  switch (api) {
  case ComputeApi::Metal:
    return "metal";
  case ComputeApi::Vulkan:
    return "vulkan";
  case ComputeApi::Cpu:
    return "cpu";
  }
  return "unknown";
}

[[nodiscard]] constexpr const char *OpName(const u8 op) noexcept {
  switch (static_cast<IrOp>(op)) {
  case IrOp::Param:
    return "param";
  case IrOp::Read:
    return "read";
  case IrOp::ReadAt:
    return "read_at";
  case IrOp::Write:
    return "write";
  case IrOp::Quantize:
    return "quantize";
  case IrOp::Add:
    return "add";
  case IrOp::Sub:
    return "sub";
  case IrOp::Mul:
    return "mul";
  case IrOp::MulWrap:
    return "mul_wrap";
  case IrOp::Min:
    return "min";
  case IrOp::Max:
    return "max";
  case IrOp::Clamp:
    return "clamp";
  case IrOp::Select:
    return "select";
  case IrOp::Eq:
    return "eq";
  case IrOp::Lt:
    return "lt";
  case IrOp::Le:
    return "le";
  case IrOp::Constant:
    return "constant";
  case IrOp::Neg:
    return "neg";
  case IrOp::Abs:
    return "abs";
  case IrOp::AbsMagnitude:
    return "abs_magnitude";
  case IrOp::Sign:
    return "sign";
  case IrOp::Ne:
    return "ne";
  case IrOp::Gt:
    return "gt";
  case IrOp::Ge:
    return "ge";
  case IrOp::PredicateNot:
    return "predicate_not";
  case IrOp::PredicateAnd:
    return "predicate_and";
  case IrOp::PredicateOr:
    return "predicate_or";
  case IrOp::BitAnd:
    return "bit_and";
  case IrOp::BitOr:
    return "bit_or";
  case IrOp::BitXor:
    return "bit_xor";
  case IrOp::BitNot:
    return "bit_not";
  case IrOp::ShlConst:
    return "shl_const";
  case IrOp::ShrLogicalConst:
    return "shr_logical_const";
  case IrOp::ShrArithmeticConst:
    return "shr_arithmetic_const";
  case IrOp::AddSat:
    return "add_sat";
  case IrOp::AddSatUnsigned:
    return "add_sat_unsigned";
  case IrOp::SubSat:
    return "sub_sat";
  case IrOp::NegPositiveFixed:
    return "neg_positive_fixed";
  case IrOp::MulFixed:
    return "mul_fixed";
  case IrOp::MulFixedScaled:
    return "mul_fixed_scaled";
  case IrOp::MulUnsignedFixed:
    return "mul_unsigned_fixed";
  case IrOp::MulAddFixed:
    return "mul_add_fixed";
  case IrOp::DivFixed:
    return "div_fixed";
  case IrOp::Recip:
    return "recip";
  case IrOp::Sqrt:
    return "sqrt";
  case IrOp::Rsqrt:
    return "reciprocal_sqrt";
  case IrOp::Sin:
    return "sin";
  case IrOp::Cos:
    return "cos";
  case IrOp::Tan:
    return "tan";
  case IrOp::Exp:
    return "exp";
  case IrOp::Log:
    return "log";
  case IrOp::Atan2:
    return "atan2";
  case IrOp::DivSigned:
    return "div_signed";
  case IrOp::DivUnsigned:
    return "div_unsigned";
  case IrOp::MinUnsigned:
    return "min_unsigned";
  case IrOp::MaxUnsigned:
    return "max_unsigned";
  case IrOp::ClampUnsigned:
    return "clamp_unsigned";
  case IrOp::LtUnsigned:
    return "lt_unsigned";
  case IrOp::LeUnsigned:
    return "le_unsigned";
  case IrOp::GtUnsigned:
    return "gt_unsigned";
  case IrOp::GeUnsigned:
    return "ge_unsigned";
  case IrOp::Index:
    return "index";
  }
  return nullptr;
}

#include <kernel/program/compute/lowering/names/ops.hpp>

[[nodiscard]] constexpr bool
BindingWidthShapeValid(const u8 kind, const u32 element_bytes,
                       const ComputeScalar scalar) noexcept {
  const u32 scalar_bytes = ScalarBytes(scalar);
  if (element_bytes == scalar_bytes) {
    return true;
  }
  return kind == kWriteBindingKind &&
         ((scalar == ComputeScalar::Lane64 && element_bytes == sizeof(u32)) ||
          (scalar == ComputeScalar::Lane32 && element_bytes == sizeof(u64)));
}

} // namespace compute_lowering_detail

} // namespace rund::kernel

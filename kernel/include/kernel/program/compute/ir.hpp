#pragma once

#include <kernel/program/compute/model.hpp>

#include <vector>

namespace rund::kernel {

enum class IrOp : u8 {
  Param,
  Read,
  Write,
  Add,
  Sub,
  Mul,
  MulWrap,
  Min,
  Max,
  Clamp,
  Select,
  Eq,
  Lt,
  Le,
  Constant,
  Neg,
  Abs,
  AbsMagnitude,
  Sign,
  Ne,
  Gt,
  Ge,
  PredicateNot,
  PredicateAnd,
  PredicateOr,
  BitAnd,
  BitOr,
  BitXor,
  BitNot,
  ShlConst,
  ShrLogicalConst,
  ShrArithmeticConst,
  AddSat,
  AddSatUnsigned,
  SubSat,
  NegPositiveFixed,
  MulFixed,
  MulFixedScaled,
  MulUnsignedFixed,
  MulAddFixed,
  DivFixed,
  Recip,
  Sqrt,
  Rsqrt,
  Sin,
  Cos,
  Tan,
  Exp,
  Log,
  Atan2,
  DivSigned,
  DivUnsigned,
  MinUnsigned,
  MaxUnsigned,
  ClampUnsigned,
  LtUnsigned,
  LeUnsigned,
  GtUnsigned,
  GeUnsigned,
  Index,
  Quantize,
  ReadAt,
  ReadUniform,
};

// Write modes are part of the canonical IR identity. Value is the ordinary
// public DSL store. The other modes are narrow, internal Flow materialization
// contracts; lowering validates their exact producer shape before treating the
// normalized result as raw storage bits in the target domain.
enum class IrWriteMode : u32 {
  Value = 0u,
  CheckedOrdinal = 1u,
  BoundaryMask = 2u,
};

[[nodiscard]] constexpr bool IrWriteModeValid(const u32 value) noexcept {
  return value <= static_cast<u32>(IrWriteMode::BoundaryMask);
}

[[nodiscard]] constexpr ComputeDomain
MergeComputeDomains(const ComputeDomain lhs, const ComputeDomain rhs) noexcept {
  const auto unknown = static_cast<ComputeDomain>(0u);
  if (lhs == unknown) {
    return rhs;
  }
  if (rhs == unknown || lhs == rhs) {
    return lhs;
  }
  if (lhs == ComputeDomain::Fixed || rhs == ComputeDomain::Fixed) {
    return unknown;
  }

  const auto pair = [lhs, rhs](const ComputeDomain first,
                               const ComputeDomain second) constexpr {
    return (lhs == first && rhs == second) || (lhs == second && rhs == first);
  };
  if (pair(ComputeDomain::I32, ComputeDomain::U32)) {
    return ComputeDomain::U32;
  }
  if (pair(ComputeDomain::I32, ComputeDomain::I64) ||
      pair(ComputeDomain::U32, ComputeDomain::I64)) {
    return ComputeDomain::I64;
  }
  if (pair(ComputeDomain::I32, ComputeDomain::U64) ||
      pair(ComputeDomain::U32, ComputeDomain::U64) ||
      pair(ComputeDomain::I64, ComputeDomain::U64)) {
    return ComputeDomain::U64;
  }
  return unknown;
}

[[nodiscard]] constexpr bool FixedOnlyOp(const IrOp op) noexcept {
  switch (op) {
  case IrOp::Quantize:
  case IrOp::NegPositiveFixed:
  case IrOp::MulFixed:
  case IrOp::MulFixedScaled:
  case IrOp::MulUnsignedFixed:
  case IrOp::MulAddFixed:
  case IrOp::DivFixed:
  case IrOp::Recip:
  case IrOp::Sqrt:
  case IrOp::Rsqrt:
  case IrOp::Sin:
  case IrOp::Cos:
  case IrOp::Tan:
  case IrOp::Exp:
  case IrOp::Log:
  case IrOp::Atan2:
    return true;
  default:
    return false;
  }
}

[[nodiscard]] constexpr IrOp
CanonicalIrOpForDomain(const IrOp op, const ComputeDomain domain) noexcept {
  if (domain != ComputeDomain::U32 && domain != ComputeDomain::U64) {
    return op;
  }
  switch (op) {
  case IrOp::Min:
    return IrOp::MinUnsigned;
  case IrOp::Max:
    return IrOp::MaxUnsigned;
  case IrOp::Clamp:
    return IrOp::ClampUnsigned;
  case IrOp::Lt:
    return IrOp::LtUnsigned;
  case IrOp::Le:
    return IrOp::LeUnsigned;
  case IrOp::Gt:
    return IrOp::GtUnsigned;
  case IrOp::Ge:
    return IrOp::GeUnsigned;
  default:
    return op;
  }
}

[[nodiscard]] constexpr bool
IrOpDomainValid(const IrOp op, const ComputeDomain domain) noexcept {
  const bool signed_domain =
      domain == ComputeDomain::I32 || domain == ComputeDomain::I64;
  const bool unsigned_domain =
      domain == ComputeDomain::U32 || domain == ComputeDomain::U64;
  const bool fixed_domain = domain == ComputeDomain::Fixed;
  if (!signed_domain && !unsigned_domain && !fixed_domain) {
    return false;
  }
  if (FixedOnlyOp(op)) {
    return fixed_domain;
  }
  switch (op) {
  case IrOp::Abs:
  case IrOp::AbsMagnitude:
  case IrOp::Sign:
  case IrOp::AddSat:
  case IrOp::SubSat:
  case IrOp::ShrArithmeticConst:
  case IrOp::Min:
  case IrOp::Max:
  case IrOp::Clamp:
  case IrOp::Lt:
  case IrOp::Le:
  case IrOp::Gt:
  case IrOp::Ge:
    return signed_domain || fixed_domain;
  case IrOp::AddSatUnsigned:
    return unsigned_domain || fixed_domain;
  case IrOp::MinUnsigned:
  case IrOp::MaxUnsigned:
  case IrOp::ClampUnsigned:
  case IrOp::LtUnsigned:
  case IrOp::LeUnsigned:
  case IrOp::GtUnsigned:
  case IrOp::GeUnsigned:
    return unsigned_domain;
  case IrOp::DivSigned:
    return signed_domain;
  case IrOp::DivUnsigned:
    return unsigned_domain;
  default:
    return true;
  }
}

struct ComputeIrNode {
  IrOp op = IrOp::Param;
  u32 lhs = 0u;
  u32 rhs = 0u;
  u32 aux = 0u;
  ComputeFixedFormat fixed_format{};
};

struct ComputeIR {
  ComputeScalar scalar = ComputeScalar::Lane32;
  ComputeDomain domain = ComputeDomain::Fixed;
  ComputeFixedFormat fixed_format{};
  u64 op_hash_hi = 0u;
  u64 op_hash_lo = 0u;
  std::vector<u8> canonical_bytes;
  bool ok = false;
  const char *reason = "compute_ir_invalid";
};

namespace compute_ir_detail {

struct ComputeIrHash {
  u64 hi = 0u;
  u64 lo = 0u;
};

[[nodiscard]] ComputeIrHash HashComputeIrCanonicalBytes(const u8 *bytes,
                                                        u64 size) noexcept;

} // namespace compute_ir_detail

} // namespace rund::kernel

#pragma once

#include <kernel/program/compute/ir.hpp>

#include <string>

namespace rund::kernel {
namespace compute_lowering_detail {

[[nodiscard]] constexpr const char *
FixedOpWidth(const ComputeScalar scalar) noexcept {
  return scalar == ComputeScalar::Lane64 ? "64" : "32";
}

[[nodiscard]] inline std::string FixedOpCall(const char *const prefix,
                                             const ComputeScalar scalar,
                                             const std::string &lhs,
                                             const std::string &rhs = {},
                                             const std::string &aux = {}) {
  std::string expr = prefix;
  expr += FixedOpWidth(scalar);
  expr += "(";
  expr += lhs;
  if (!rhs.empty()) {
    expr += ", ";
    expr += rhs;
  }
  if (!aux.empty()) {
    expr += ", ";
    expr += aux;
  }
  expr += ")";
  return expr;
}

[[nodiscard]] inline std::string
FixedLaneOpCall(const char *const prefix, const ComputeScalar scalar,
                const std::string &lhs, const std::string &rhs = {},
                const std::string &aux = {}) {
  std::string name = prefix;
  name += "Lane";
  return FixedOpCall(name.c_str(), scalar, lhs, rhs, aux);
}

[[nodiscard]] inline std::string
FixedOpExpr(const ComputeScalar scalar, const IrOp op, const std::string &lhs,
            const std::string &rhs, const std::string &) {
  switch (op) {
  case IrOp::AddSat:
    return FixedOpCall("RundAddSat", scalar, lhs, rhs);
  case IrOp::AddSatUnsigned:
    return FixedOpCall("RundAddSatUnsigned", scalar, lhs, rhs);
  case IrOp::SubSat:
    return FixedOpCall("RundSubSat", scalar, lhs, rhs);
  case IrOp::NegPositiveFixed:
    return FixedLaneOpCall("RundNegPositiveFixed", scalar, lhs);
  case IrOp::MulFixed:
    return FixedLaneOpCall("RundMulFixed", scalar, lhs, rhs);
  case IrOp::MulFixedScaled:
    return FixedOpCall("RundMulFixedScaled", scalar, lhs, rhs);
  case IrOp::Sin:
    return FixedOpCall("RundSin", scalar, lhs);
  case IrOp::Cos:
    return FixedOpCall("RundCos", scalar, lhs);
  case IrOp::Exp:
    return FixedOpCall("RundExp", scalar, lhs);
  case IrOp::Log:
    return FixedOpCall("RundLog", scalar, lhs);
  case IrOp::Atan2:
    return FixedOpCall("RundAtan2", scalar, lhs, rhs);
  default:
    return {};
  }
}

} // namespace compute_lowering_detail
} // namespace rund::kernel

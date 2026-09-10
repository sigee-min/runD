#pragma once

#include <rund/compute/abi/state.hpp>
#include <rund/compute/fixed.hpp>
#include <cstdint>
#include <memory>

namespace rund::compute::detail {

enum class ExprOp : unsigned char {
  Input,
  Constant,
  Index,
  Add,
  Subtract,
  Multiply,
  MultiplyWrap,
  Divide,
  Negate,
  BitAnd,
  BitOr,
  BitXor,
  Min,
  Max,
  Clamp,
  Equal,
  NotEqual,
  Less,
  LessEqual,
  Greater,
  GreaterEqual,
  PredicateNot,
  PredicateAnd,
  PredicateOr,
  Mask,
  Select,
  Abs,
  AbsMagnitude,
  Sign,
  BitNot,
  AddSat,
  AddSatUnsigned,
  SubSat,
  NegPositiveFixed,
  MulFixed,
  MulFixedScaled,
  MulUnsignedFixed,
  MulAddFixed,
  Reciprocal,
  Sqrt,
  Rsqrt,
  Sin,
  Cos,
  Tan,
  Exp,
  Log,
  Atan2,
  ShiftLeft,
  ShiftRightLogical,
  ShiftRightArithmetic,
  Quantize,
  CheckedOrdinal,
  BoundaryMask,
};
struct ExprRef final {
  std::shared_ptr<ExprState> state;
  std::uint32_t node{};
  Type type{Type::I32};
  FixedFormat fixed_format{};
};

} // namespace rund::compute::detail

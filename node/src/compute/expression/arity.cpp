#include "state.hpp"

namespace rund::compute::detail {

std::uint8_t expr_arity(const ExprOp operation) noexcept {
  switch (operation) {
  case ExprOp::Input:
  case ExprOp::Constant:
  case ExprOp::Index:
    return 0u;
  case ExprOp::Negate:
  case ExprOp::Abs:
  case ExprOp::AbsMagnitude:
  case ExprOp::Sign:
  case ExprOp::BitNot:
  case ExprOp::NegPositiveFixed:
  case ExprOp::Reciprocal:
  case ExprOp::Sqrt:
  case ExprOp::Rsqrt:
  case ExprOp::Sin:
  case ExprOp::Cos:
  case ExprOp::Tan:
  case ExprOp::Exp:
  case ExprOp::Log:
  case ExprOp::ShiftLeft:
  case ExprOp::ShiftRightLogical:
  case ExprOp::ShiftRightArithmetic:
  case ExprOp::PredicateNot:
  case ExprOp::Mask:
  case ExprOp::Quantize:
  case ExprOp::CheckedOrdinal:
  case ExprOp::BoundaryMask:
    return 1u;
  case ExprOp::Add:
  case ExprOp::Subtract:
  case ExprOp::Multiply:
  case ExprOp::MultiplyWrap:
  case ExprOp::Divide:
  case ExprOp::BitAnd:
  case ExprOp::BitOr:
  case ExprOp::BitXor:
  case ExprOp::Min:
  case ExprOp::Max:
  case ExprOp::Equal:
  case ExprOp::NotEqual:
  case ExprOp::Less:
  case ExprOp::LessEqual:
  case ExprOp::Greater:
  case ExprOp::GreaterEqual:
  case ExprOp::PredicateAnd:
  case ExprOp::PredicateOr:
  case ExprOp::AddSat:
  case ExprOp::AddSatUnsigned:
  case ExprOp::SubSat:
  case ExprOp::MulFixed:
  case ExprOp::MulFixedScaled:
  case ExprOp::MulUnsignedFixed:
  case ExprOp::Atan2:
    return 2u;
  case ExprOp::Clamp:
  case ExprOp::Select:
  case ExprOp::MulAddFixed:
    return 3u;
  }
  return InvalidArity;
}

bool supported(const ExprOp operation) noexcept {
  return expr_arity(operation) != InvalidArity;
}

} // namespace rund::compute::detail

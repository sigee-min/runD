#include "model.hpp"

#include "../../type.hpp"

namespace rund::compute::detail {

std::optional<kernel::IrOp> binary_op(const ExprOp operation,
                                      const Type type) noexcept {
  switch (operation) {
  case ExprOp::Add:
    return kernel::IrOp::Add;
  case ExprOp::Subtract:
    return kernel::IrOp::Sub;
  case ExprOp::Multiply:
    return kernel::IrOp::Mul;
  case ExprOp::MultiplyWrap:
    return kernel::IrOp::MulWrap;
  case ExprOp::Divide:
    switch (type) {
    case Type::I32:
    case Type::I64:
      return kernel::IrOp::DivSigned;
    case Type::U32:
    case Type::U64:
      return kernel::IrOp::DivUnsigned;
    case Type::FixedLane32:
    case Type::FixedLane64:
      return kernel::IrOp::DivFixed;
    }
    return std::nullopt;
  case ExprOp::BitAnd:
    return kernel::IrOp::BitAnd;
  case ExprOp::BitOr:
    return kernel::IrOp::BitOr;
  case ExprOp::BitXor:
    return kernel::IrOp::BitXor;
  case ExprOp::Min:
    return kernel::CanonicalIrOpForDomain(kernel::IrOp::Min, type_domain(type));
  case ExprOp::Max:
    return kernel::CanonicalIrOpForDomain(kernel::IrOp::Max, type_domain(type));
  case ExprOp::Equal:
    return kernel::IrOp::Eq;
  case ExprOp::NotEqual:
    return kernel::IrOp::Ne;
  case ExprOp::Less:
    return kernel::CanonicalIrOpForDomain(kernel::IrOp::Lt, type_domain(type));
  case ExprOp::LessEqual:
    return kernel::CanonicalIrOpForDomain(kernel::IrOp::Le, type_domain(type));
  case ExprOp::Greater:
    return kernel::CanonicalIrOpForDomain(kernel::IrOp::Gt, type_domain(type));
  case ExprOp::GreaterEqual:
    return kernel::CanonicalIrOpForDomain(kernel::IrOp::Ge, type_domain(type));
  case ExprOp::PredicateAnd:
    return kernel::IrOp::PredicateAnd;
  case ExprOp::PredicateOr:
    return kernel::IrOp::PredicateOr;
  case ExprOp::AddSat:
    return kernel::IrOp::AddSat;
  case ExprOp::AddSatUnsigned:
    return kernel::IrOp::AddSatUnsigned;
  case ExprOp::SubSat:
    return kernel::IrOp::SubSat;
  case ExprOp::MulFixed:
    return kernel::IrOp::MulFixed;
  case ExprOp::MulFixedScaled:
    return kernel::IrOp::MulFixedScaled;
  case ExprOp::MulUnsignedFixed:
    return kernel::IrOp::MulUnsignedFixed;
  case ExprOp::Atan2:
    return kernel::IrOp::Atan2;
  default:
    return std::nullopt;
  }
}

std::optional<kernel::IrOp> unary_op(const ExprOp operation) noexcept {
  switch (operation) {
  case ExprOp::Negate:
    return kernel::IrOp::Neg;
  case ExprOp::Abs:
    return kernel::IrOp::Abs;
  case ExprOp::AbsMagnitude:
    return kernel::IrOp::AbsMagnitude;
  case ExprOp::Sign:
    return kernel::IrOp::Sign;
  case ExprOp::BitNot:
    return kernel::IrOp::BitNot;
  case ExprOp::NegPositiveFixed:
    return kernel::IrOp::NegPositiveFixed;
  case ExprOp::Reciprocal:
    return kernel::IrOp::Recip;
  case ExprOp::Sqrt:
    return kernel::IrOp::Sqrt;
  case ExprOp::Rsqrt:
    return kernel::IrOp::Rsqrt;
  case ExprOp::Sin:
    return kernel::IrOp::Sin;
  case ExprOp::Cos:
    return kernel::IrOp::Cos;
  case ExprOp::Tan:
    return kernel::IrOp::Tan;
  case ExprOp::Exp:
    return kernel::IrOp::Exp;
  case ExprOp::Log:
    return kernel::IrOp::Log;
  default:
    return std::nullopt;
  }
}

} // namespace rund::compute::detail

#include "contract/program/compute/dsl/reject/model.hpp"
#include "test/assert.hpp"

#include <array>

namespace program_compute_contract::dsl_reject {
namespace {

int FixedStorage() {
  constexpr std::array unary_ops{
      rund::kernel::IrOp::BitNot, rund::kernel::IrOp::NegPositiveFixed,
      rund::kernel::IrOp::Recip,  rund::kernel::IrOp::Sqrt,
      rund::kernel::IrOp::Rsqrt,  rund::kernel::IrOp::Sin,
      rund::kernel::IrOp::Cos,    rund::kernel::IrOp::Tan,
      rund::kernel::IrOp::Exp,    rund::kernel::IrOp::Log,
  };
  for (const auto op : unary_ops) {
    TEST_ASSERT(Rejects(FixedOp(op, Arity::Unary, false, true),
                        "compute_fixed_quantize_required"));
  }
  TEST_ASSERT(
      Rejects(FixedOp(rund::kernel::IrOp::BitNot, Arity::Unary, true, true),
              "compute_fixed_quantize_required"));

  constexpr std::array binary_ops{
      rund::kernel::IrOp::MulWrap,        rund::kernel::IrOp::BitAnd,
      rund::kernel::IrOp::BitOr,          rund::kernel::IrOp::BitXor,
      rund::kernel::IrOp::AddSat,         rund::kernel::IrOp::AddSatUnsigned,
      rund::kernel::IrOp::SubSat,         rund::kernel::IrOp::MulFixed,
      rund::kernel::IrOp::MulFixedScaled, rund::kernel::IrOp::MulUnsignedFixed,
      rund::kernel::IrOp::DivFixed,       rund::kernel::IrOp::Atan2,
  };
  for (const auto op : binary_ops) {
    TEST_ASSERT(Rejects(FixedOp(op, Arity::Binary, false, true),
                        "compute_fixed_quantize_required"));
  }
  TEST_ASSERT(
      Rejects(FixedOp(rund::kernel::IrOp::BitAnd, Arity::Binary, true, true),
              "compute_fixed_quantize_required"));

  for (const auto op :
       {rund::kernel::IrOp::ShlConst, rund::kernel::IrOp::ShrLogicalConst,
        rund::kernel::IrOp::ShrArithmeticConst}) {
    TEST_ASSERT(Rejects(FixedOp(op, Arity::Shift, false, true),
                        "compute_fixed_quantize_required"));
  }
  return 0;
}

int IntegerDivide() {
  for (const auto op :
       {rund::kernel::IrOp::DivSigned, rund::kernel::IrOp::DivUnsigned}) {
    TEST_ASSERT(Rejects(FixedOp(op, Arity::Binary)));
  }
  return 0;
}

} // namespace

int Storage() {
  if (FixedStorage() != 0) {
    return 1;
  }
  return IntegerDivide();
}

} // namespace program_compute_contract::dsl_reject

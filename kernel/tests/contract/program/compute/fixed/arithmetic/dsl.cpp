#include "contract/program/compute/fixed/arithmetic/local.hpp"

namespace program_compute_contract {
namespace {

using namespace lowering_support;

int test_compute_fixed_arithmetic_dsl_admits_fixed_ops() {
  const auto fixed_lane32 = BuildFixedLane32ArithmeticOps();
  const auto fixed_lane64 = BuildFixedLane64ArithmeticOps();

  TEST_ASSERT(fixed_lane32.ok());
  TEST_ASSERT(fixed_lane64.ok());
  TEST_ASSERT(fixed_lane32.ir().scalar == rund::kernel::ComputeScalar::Lane32);
  TEST_ASSERT(fixed_lane64.ir().scalar == rund::kernel::ComputeScalar::Lane64);
  return 0;
}

template <unsigned IntegerBits, unsigned FractionBits, class Op>
int test_declared_multiply_ir(const Op &op) {
  using rund::kernel::IrOp;
  TEST_ASSERT(op.ok());
  const auto parsed =
      rund::kernel::compute_lowering_detail::ParseComputeIR(op.ir());
  TEST_ASSERT(parsed.ok);
  bool saw_signed = false;
  bool saw_scaled = false;
  bool saw_unsigned = false;
  bool saw_mul_add = false;
  for (const auto &node : parsed.nodes) {
    const auto operation = static_cast<IrOp>(node.op);
    if (operation == IrOp::MulFixed || operation == IrOp::MulFixedScaled ||
        operation == IrOp::MulUnsignedFixed) {
      TEST_ASSERT(node.fixed_format.integer_bits == IntegerBits);
      TEST_ASSERT(node.fixed_format.fraction_bits == FractionBits);
      TEST_ASSERT(node.fixed_format.rounding ==
                  rund::kernel::ComputeRounding::NearestEven);
      TEST_ASSERT(node.fixed_format.overflow ==
                  rund::kernel::ComputeOverflow::Saturate);
      saw_signed = saw_signed || operation == IrOp::MulFixed;
      saw_scaled = saw_scaled || operation == IrOp::MulFixedScaled;
      saw_unsigned = saw_unsigned || operation == IrOp::MulUnsignedFixed;
    }
    if (operation != IrOp::MulAddFixed) {
      continue;
    }
    saw_mul_add = true;
    TEST_ASSERT(node.fixed_format.integer_bits == IntegerBits * 2u);
    TEST_ASSERT(node.fixed_format.fraction_bits == FractionBits * 2u);
  }
  TEST_ASSERT(saw_signed);
  TEST_ASSERT(saw_scaled);
  TEST_ASSERT(saw_unsigned);
  TEST_ASSERT(saw_mul_add);
  return 0;
}

int test_compute_fixed_arithmetic_ir_preserves_declared_multiply_formats() {
  const auto fixed16 = BuildFixed16_16DeclaredMultiplyOps();
  const auto fixed44 = BuildFixed20_44DeclaredMultiplyOps();
  if (test_declared_multiply_ir<16u, 16u>(fixed16) != 0) {
    return 1;
  }
  return test_declared_multiply_ir<20u, 44u>(fixed44);
}

} // namespace

int RunComputeFixedArithmeticDslContract() {
  if (test_compute_fixed_arithmetic_dsl_admits_fixed_ops() != 0) {
    return 1;
  }
  return test_compute_fixed_arithmetic_ir_preserves_declared_multiply_formats();
}

} // namespace program_compute_contract

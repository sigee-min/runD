#include "contract/program/compute/fixed/arithmetic/local.hpp"

namespace program_compute_contract {
namespace {

using namespace lowering_support;

int test_compute_fixed_arithmetic_malformed_nodes_reject() {
  const rund::kernel::LoweringArtifact binary_aux =
      rund::kernel::LowerComputeIR(IrFromBytes(FixedArithmeticBinaryIrBytes(
                                       rund::kernel::IrOp::AddSat, 1u, 2u, 3u)),
                                   rund::kernel::ComputeApi::Metal);
  TEST_ASSERT(!binary_aux.ok);
  TEST_ASSERT(std::string_view{binary_aux.reason} == "compute_ir_node_invalid");

  const rund::kernel::LoweringArtifact unary_arity =
      rund::kernel::LowerComputeIR(IrFromBytes(WrongUnaryArityIrBytes(
                                       rund::kernel::IrOp::NegPositiveFixed)),
                                   rund::kernel::ComputeApi::Metal);
  TEST_ASSERT(!unary_arity.ok);
  TEST_ASSERT(std::string_view{unary_arity.reason} ==
              "compute_ir_node_invalid");

  const rund::kernel::LoweringArtifact ternary_ref =
      rund::kernel::LowerComputeIR(
          IrFromBytes(FixedArithmeticTernaryIrBytes(
              rund::kernel::IrOp::MulAddFixed, 1u, 2u, 9u)),
          rund::kernel::ComputeApi::Metal);
  TEST_ASSERT(!ternary_ref.ok);
  TEST_ASSERT(std::string_view{ternary_ref.reason} ==
              "compute_ir_node_invalid");
  return 0;
}

} // namespace

int RunComputeFixedArithmeticRejectContract() {
  return test_compute_fixed_arithmetic_malformed_nodes_reject();
}

} // namespace program_compute_contract

#include "test/assert.hpp"

#include <kernel/program/compute/ir.hpp>

#include <string_view>

namespace program_compute_contract {

namespace {

int test_compute_ir_default_is_invalid() {
  const rund::kernel::ComputeIR ir{};

  TEST_ASSERT(ir.scalar == rund::kernel::ComputeScalar::Lane32);
  TEST_ASSERT(ir.op_hash_hi == 0u);
  TEST_ASSERT(ir.op_hash_lo == 0u);
  TEST_ASSERT(!ir.ok);
  TEST_ASSERT(std::string_view{ir.reason} == "compute_ir_invalid");
  return 0;
}

int test_compute_ir_closed_node_model_defaults_to_param() {
  const rund::kernel::ComputeIrNode node{};

  TEST_ASSERT(node.op == rund::kernel::IrOp::Param);
  TEST_ASSERT(node.lhs == 0u);
  TEST_ASSERT(node.rhs == 0u);
  TEST_ASSERT(node.aux == 0u);
  return 0;
}

}  // namespace

int RunComputeIrContract() {
  if (test_compute_ir_default_is_invalid() != 0) {
    return 1;
  }
  if (test_compute_ir_closed_node_model_defaults_to_param() != 0) {
    return 1;
  }
  return 0;
}

}  // namespace program_compute_contract

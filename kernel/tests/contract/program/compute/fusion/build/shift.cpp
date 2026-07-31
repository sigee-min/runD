#include "model.hpp"

namespace program_compute_contract::fusion_build_contract {
namespace {

int test_compute_fusion_remaps_bit_shift_inputs() {
  const rund::compute_dsl::ComputeOp first = BuildAddFiveOp();
  const rund::compute_dsl::ComputeOp second = BuildBitShiftConsumerOp();
  TEST_ASSERT(first.ok());
  TEST_ASSERT(second.ok());

  const rund::kernel::ComputeFusedMapChainIR fused =
      BuildPair(first, second, rund::kernel::ComputeApi::Metal);
  TEST_ASSERT(fused.ok);

  const rund::kernel::compute_lowering_detail::ParsedIR parsed =
      rund::kernel::compute_lowering_detail::ParseComputeIR(fused.ir);
  TEST_ASSERT(parsed.ok);

  const rund::kernel::u32 producer_node = FirstProducerOutputNode(parsed);
  TEST_ASSERT(producer_node != 0u);

  bool saw_shl = false;
  bool saw_shr_logical = false;
  bool saw_shr_arithmetic = false;
  for (const auto &node : parsed.nodes) {
    const auto op = static_cast<rund::kernel::IrOp>(node.op);
    if (!rund::kernel::compute_lowering_detail::ConstShiftOp(op)) {
      continue;
    }
    TEST_ASSERT(node.lhs == producer_node);
    TEST_ASSERT(node.rhs == 0u);
    if (op == rund::kernel::IrOp::ShlConst) {
      saw_shl = node.aux == 3u;
    } else if (op == rund::kernel::IrOp::ShrLogicalConst) {
      saw_shr_logical = node.aux == 5u;
    } else if (op == rund::kernel::IrOp::ShrArithmeticConst) {
      saw_shr_arithmetic = node.aux == 7u;
    }
  }
  TEST_ASSERT(saw_shl);
  TEST_ASSERT(saw_shr_logical);
  TEST_ASSERT(saw_shr_arithmetic);

  const rund::kernel::LoweringArtifact artifact =
      rund::kernel::LowerComputeIR(fused.ir, rund::kernel::ComputeApi::Metal);
  TEST_ASSERT(artifact.ok);
  return 0;
}

} // namespace

int RunShift() { return test_compute_fusion_remaps_bit_shift_inputs(); }

} // namespace program_compute_contract::fusion_build_contract

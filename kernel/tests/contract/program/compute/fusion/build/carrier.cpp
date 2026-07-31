#include "model.hpp"

namespace program_compute_contract::fusion_build_contract {
namespace {

template <rund::kernel::IrWriteMode WriteMode>
[[nodiscard]] int CheckFusedCarrierMode() {
  using Mode = rund::compute_dsl::detail::ScalarMode;
  const rund::compute_dsl::ComputeOp first =
      BuildCarrierProducerOp<Mode::U32>();
  const rund::compute_dsl::ComputeOp second =
      BuildCarrierConsumerOp<WriteMode>();
  TEST_ASSERT(first.ok());
  TEST_ASSERT(second.ok());
  for (const rund::kernel::ComputeApi api :
       {rund::kernel::ComputeApi::Cpu, rund::kernel::ComputeApi::Metal,
        rund::kernel::ComputeApi::Vulkan}) {
    const rund::kernel::ComputeFusedMapChainIR fused =
        BuildPair(first, second, api);
    TEST_ASSERT(fused.ok);
    TEST_ASSERT(fused.fusion.ok);
    TEST_ASSERT(fused.fusion.original_node_count == 2u);
    TEST_ASSERT(fused.fusion.fused_node_count == 1u);
    TEST_ASSERT(fused.fusion.rejected_edge_count == 0u);
    const auto parsed =
        rund::kernel::compute_lowering_detail::ParseComputeIR(fused.ir);
    TEST_ASSERT(parsed.ok);
    rund::kernel::u32 add = 0u;
    rund::kernel::u32 predicate = 0u;
    rund::kernel::u32 select = 0u;
    const rund::kernel::compute_lowering_detail::ParsedNode *write = nullptr;
    for (std::size_t index = 0u; index < parsed.nodes.size(); ++index) {
      const auto op = static_cast<rund::kernel::IrOp>(parsed.nodes[index].op);
      if (op == rund::kernel::IrOp::Add) {
        add = static_cast<rund::kernel::u32>(index + 1u);
      } else if (op == rund::kernel::IrOp::LeUnsigned ||
                 op == rund::kernel::IrOp::Ne) {
        predicate = static_cast<rund::kernel::u32>(index + 1u);
      } else if (op == rund::kernel::IrOp::Select) {
        select = static_cast<rund::kernel::u32>(index + 1u);
      } else if (op == rund::kernel::IrOp::Write) {
        write = &parsed.nodes[index];
      }
    }
    TEST_ASSERT(add != 0u && predicate != 0u && select != 0u &&
                write != nullptr);
    TEST_ASSERT(write->rhs == static_cast<rund::kernel::u32>(WriteMode));
    TEST_ASSERT(write->lhs == select);
    TEST_ASSERT(parsed.nodes[select - 1u].lhs == predicate);
    TEST_ASSERT(parsed.nodes[predicate - 1u].lhs == add);
    if constexpr (WriteMode == rund::kernel::IrWriteMode::CheckedOrdinal) {
      TEST_ASSERT(parsed.nodes[select - 1u].rhs == add);
      TEST_ASSERT(rund::kernel::ComputeFixedFormatAbsent(write->fixed_format));
    } else {
      TEST_ASSERT(write->fixed_format == kCarrierFormat16x16);
    }
    const rund::kernel::LoweringArtifact artifact =
        rund::kernel::LowerComputeIR(fused.ir, api);
    TEST_ASSERT(artifact.ok);
    TEST_ASSERT(!artifact.source_text.empty());
  }
  return 0;
}

} // namespace

int RunCarrier() {
  if (CheckFusedCarrierMode<rund::kernel::IrWriteMode::CheckedOrdinal>() != 0) {
    return 1;
  }
  return CheckFusedCarrierMode<rund::kernel::IrWriteMode::BoundaryMask>();
}

} // namespace program_compute_contract::fusion_build_contract

#include "model.hpp"

namespace program_compute_contract::fusion_build_contract {
namespace {

int test_compute_fusion_keeps_terminal_multiwrite_outputs() {
  const rund::compute_dsl::ComputeOp first = BuildAddFiveOp();
  const rund::compute_dsl::ComputeOp second = BuildMultiWriteOp();
  TEST_ASSERT(first.ok());
  TEST_ASSERT(second.ok());

  const rund::kernel::GraphBufferRef first_buffers[2] = {
      {.logical_id = 11u, .role = rund::kernel::BufferRole::Read},
      {.logical_id = 21u, .role = rund::kernel::BufferRole::Write},
  };
  const rund::kernel::GraphBufferRef second_buffers[4] = {
      {.logical_id = 21u, .role = rund::kernel::BufferRole::Read},
      {.logical_id = 31u, .role = rund::kernel::BufferRole::Write},
      {.logical_id = 32u, .role = rund::kernel::BufferRole::Write},
      {.logical_id = 33u, .role = rund::kernel::BufferRole::Write},
  };
  const rund::kernel::GraphNode nodes[2] = {
      {.op_hash_hi = first.ir().op_hash_hi,
       .op_hash_lo = first.ir().op_hash_lo,
       .buffers = first_buffers,
       .buffer_count = 2u,
       .element_count = 4u},
      {.op_hash_hi = second.ir().op_hash_hi,
       .op_hash_lo = second.ir().op_hash_lo,
       .buffers = second_buffers,
       .buffer_count = 4u,
       .element_count = 4u},
  };
  const rund::kernel::Graph graph{
      .nodes = nodes,
      .node_count = 2u,
      .scalar = rund::kernel::ComputeScalar::Lane32,
      .domain = first.ir().domain,
      .fixed_format = first.ir().fixed_format,
  };
  const rund::kernel::FusionNodePolicy fusion_nodes[2] = {
      PolicyNode(first.ir()),
      PolicyNode(second.ir()),
  };
  const rund::kernel::FusionPolicy policy{
      .nodes = fusion_nodes,
      .node_count = 2u,
  };
  const rund::kernel::ComputeIR chain[2] = {first.ir(), second.ir()};

  for (const rund::kernel::ComputeApi api :
       {rund::kernel::ComputeApi::Cpu, rund::kernel::ComputeApi::Metal,
        rund::kernel::ComputeApi::Vulkan}) {
    auto admitted = rund::kernel::compute_lowering_detail::
        BuildAdmittedFusedComputeMapChainIR(chain, 2u, graph, policy, api);
    TEST_ASSERT(admitted.value.ok);
    TEST_ASSERT(admitted.value.fusion.ok);
    TEST_ASSERT(admitted.value.fusion.fused_node_count == 1u);
    TEST_ASSERT(admitted.value.metadata.read_count == 1u);
    TEST_ASSERT(admitted.value.metadata.write_count == 3u);
    const auto parsed = rund::kernel::compute_lowering_detail::ParseComputeIR(
        admitted.value.ir);
    TEST_ASSERT(parsed.ok);
    std::size_t write_nodes = 0u;
    for (const auto &node : parsed.nodes) {
      write_nodes +=
          node.op == static_cast<rund::kernel::u8>(rund::kernel::IrOp::Write);
    }
    TEST_ASSERT(write_nodes == 3u);
    const rund::kernel::LoweringArtifact artifact =
        rund::kernel::LowerComputeIR(admitted.value.ir, api);
    TEST_ASSERT(artifact.ok);
  }
  return 0;
}

} // namespace

int RunOutput() {
  return test_compute_fusion_keeps_terminal_multiwrite_outputs();
}

} // namespace program_compute_contract::fusion_build_contract

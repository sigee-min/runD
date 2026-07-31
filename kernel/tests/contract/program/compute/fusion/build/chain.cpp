#include "model.hpp"

namespace program_compute_contract::fusion_build_contract {
namespace {

int test_compute_fusion_builds_one_artifact_for_arbitrary_chain() {
  const rund::compute_dsl::ComputeOp first = BuildAddFiveOp();
  const rund::compute_dsl::ComputeOp second = BuildMulThreeOp();
  const rund::compute_dsl::ComputeOp third = BuildAddFiveOp();
  TEST_ASSERT(first.ok());
  TEST_ASSERT(second.ok());
  TEST_ASSERT(third.ok());

  const rund::kernel::GraphBufferRef first_buffers[2] = {
      {.logical_id = 11u, .role = rund::kernel::BufferRole::Read},
      {.logical_id = 21u, .role = rund::kernel::BufferRole::Write},
  };
  const rund::kernel::GraphBufferRef second_buffers[2] = {
      {.logical_id = 21u, .role = rund::kernel::BufferRole::Read},
      {.logical_id = 31u, .role = rund::kernel::BufferRole::Write},
  };
  const rund::kernel::GraphBufferRef third_buffers[2] = {
      {.logical_id = 31u, .role = rund::kernel::BufferRole::Read},
      {.logical_id = 41u, .role = rund::kernel::BufferRole::Write},
  };
  const rund::kernel::GraphNode nodes[3] = {
      {.op_hash_hi = first.ir().op_hash_hi,
       .op_hash_lo = first.ir().op_hash_lo,
       .buffers = first_buffers,
       .buffer_count = 2u,
       .element_count = 4u},
      {.op_hash_hi = second.ir().op_hash_hi,
       .op_hash_lo = second.ir().op_hash_lo,
       .buffers = second_buffers,
       .buffer_count = 2u,
       .element_count = 4u},
      {.op_hash_hi = third.ir().op_hash_hi,
       .op_hash_lo = third.ir().op_hash_lo,
       .buffers = third_buffers,
       .buffer_count = 2u,
       .element_count = 4u},
  };
  const rund::kernel::Graph graph{
      .nodes = nodes,
      .node_count = 3u,
      .scalar = rund::kernel::ComputeScalar::Lane32,
      .domain = first.ir().domain,
      .fixed_format = first.ir().fixed_format,
  };
  const rund::kernel::FusionNodePolicy fusion_nodes[3] = {
      PolicyNode(first.ir()),
      PolicyNode(second.ir()),
      PolicyNode(third.ir()),
  };
  const rund::kernel::FusionPolicy policy{
      .nodes = fusion_nodes,
      .node_count = 3u,
  };
  const rund::kernel::ComputeIR chain[3] = {first.ir(), second.ir(),
                                            third.ir()};

  for (const rund::kernel::ComputeApi api :
       {rund::kernel::ComputeApi::Cpu, rund::kernel::ComputeApi::Metal,
        rund::kernel::ComputeApi::Vulkan}) {
    auto admitted = rund::kernel::compute_lowering_detail::
        BuildAdmittedFusedComputeMapChainIR(chain, 3u, graph, policy, api);
    TEST_ASSERT(admitted.value.ok);
    TEST_ASSERT(admitted.value.fusion.ok);
    TEST_ASSERT(admitted.value.fusion.original_node_count == 3u);
    TEST_ASSERT(admitted.value.fusion.fused_node_count == 1u);
    TEST_ASSERT(admitted.value.fusion.rejected_edge_count == 0u);
    TEST_ASSERT(admitted.source_parse_count == 3u);
    TEST_ASSERT(admitted.input.ok);
    TEST_ASSERT(admitted.input.parse_count == 0u);
    TEST_ASSERT(admitted.value.metadata.ok);
    TEST_ASSERT(admitted.value.metadata.read_count == 1u);
    TEST_ASSERT(admitted.value.metadata.write_count == 1u);
    TEST_ASSERT(admitted.value.metadata.param_storage.size() == 12u);

    const auto parsed = rund::kernel::compute_lowering_detail::ParseComputeIR(
        admitted.value.ir);
    TEST_ASSERT(parsed.ok);
    rund::kernel::u32 add_count = 0u;
    rund::kernel::u32 multiply_count = 0u;
    for (const auto &node : parsed.nodes) {
      const auto op = static_cast<rund::kernel::IrOp>(node.op);
      add_count += op == rund::kernel::IrOp::Add ? 1u : 0u;
      multiply_count += op == rund::kernel::IrOp::Mul ? 1u : 0u;
    }
    TEST_ASSERT(add_count == 2u);
    TEST_ASSERT(multiply_count == 1u);

    auto emitted = rund::kernel::compute_lowering_detail::
        EmitGeneratedRetainedComputeArtifact(std::move(admitted.value.ir),
                                             std::move(admitted.value.metadata),
                                             std::move(admitted.input));
    TEST_ASSERT(emitted.artifact.ok);
    TEST_ASSERT(emitted.emission_count == 1u);
    TEST_ASSERT(emitted.parse_count() == 0u);
    TEST_ASSERT(emitted.artifact.canonical_ir_bytes.empty());
    TEST_ASSERT(!emitted.artifact.source_text.empty());
  }
  return 0;
}

} // namespace

int RunChain() {
  return test_compute_fusion_builds_one_artifact_for_arbitrary_chain();
}

} // namespace program_compute_contract::fusion_build_contract

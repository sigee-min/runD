#include "model.hpp"

namespace program_compute_contract::fusion_build_contract {
namespace {

int test_compute_fusion_builds_checked_fused_ir_for_two_map_chain() {
  const rund::compute_dsl::ComputeOp first = BuildAddFiveOp();
  const rund::compute_dsl::ComputeOp second = BuildMulThreeOp();
  TEST_ASSERT(first.ok());
  TEST_ASSERT(second.ok());

  const Pair pair{first, second};

  rund::kernel::ComputeIR first_key_forged = first.ir();
  ++first_key_forged.op_hash_lo;
  const rund::kernel::ComputeIR *key_chain[2] = {&first_key_forged,
                                                 &second.ir()};
  const auto key_rejected = rund::kernel::compute_lowering_detail::
      BuildAdmittedFusedComputeMapChainIR(key_chain, 2u, pair.graph,
                                          pair.policy,
                                          rund::kernel::ComputeApi::Metal);
  TEST_ASSERT(!key_rejected.value.ok);
  TEST_ASSERT(std::string_view{key_rejected.value.reason} ==
              "compute_ir_hash_mismatch");
  TEST_ASSERT(key_rejected.source_parse_count == 0u);

  rund::kernel::ComputeIR second_payload_forged = second.ir();
  second_payload_forged.canonical_bytes.push_back(0xffu);
  const rund::kernel::ComputeIR *payload_chain[2] = {&first.ir(),
                                                     &second_payload_forged};
  const auto payload_rejected = rund::kernel::compute_lowering_detail::
      BuildAdmittedFusedComputeMapChainIR(payload_chain, 2u, pair.graph,
                                          pair.policy,
                                          rund::kernel::ComputeApi::Metal);
  TEST_ASSERT(!payload_rejected.value.ok);
  TEST_ASSERT(std::string_view{payload_rejected.value.reason} ==
              "compute_ir_hash_mismatch");
  TEST_ASSERT(payload_rejected.source_parse_count == 1u);

  const rund::kernel::ComputeIR *admitted_chain[2] = {&first.ir(),
                                                      &second.ir()};
  auto admitted = rund::kernel::compute_lowering_detail::
      BuildAdmittedFusedComputeMapChainIR(admitted_chain, 2u, pair.graph,
                                          pair.policy,
                                          rund::kernel::ComputeApi::Metal);
  rund::kernel::ComputeFusedMapChainIR &fused = admitted.value;
  TEST_ASSERT(fused.ok);
  TEST_ASSERT(admitted.source_parse_count == 2u);
  TEST_ASSERT(admitted.input.ok);
  TEST_ASSERT(admitted.input.parse_count == 0u);
  TEST_ASSERT(std::string_view{fused.reason} == "ok");
  TEST_ASSERT(fused.fusion.ok);
  TEST_ASSERT(fused.fusion.original_node_count == 2u);
  TEST_ASSERT(fused.fusion.fused_node_count == 1u);
  TEST_ASSERT(fused.fusion.rejected_edge_count == 0u);
  TEST_ASSERT(fused.ir.ok);
  TEST_ASSERT(fused.ir.op_hash_hi != first.ir().op_hash_hi ||
              fused.ir.op_hash_lo != first.ir().op_hash_lo);

  const auto parsed =
      rund::kernel::compute_lowering_detail::ParseComputeIR(fused.ir);
  TEST_ASSERT(parsed.ok);
  TEST_ASSERT(!parsed.nodes.empty());
  const auto &write = parsed.nodes.back();
  TEST_ASSERT(write.op ==
              static_cast<rund::kernel::u8>(rund::kernel::IrOp::Write));
  TEST_ASSERT(write.lhs != 0u && write.lhs <= parsed.nodes.size());
  const auto &quantize = parsed.nodes[write.lhs - 1u];
  TEST_ASSERT(quantize.op ==
              static_cast<rund::kernel::u8>(rund::kernel::IrOp::Quantize));
  TEST_ASSERT(quantize.lhs != 0u && quantize.lhs <= parsed.nodes.size());
  TEST_ASSERT(parsed.nodes[quantize.lhs - 1u].op ==
              static_cast<rund::kernel::u8>(rund::kernel::IrOp::Mul));

  TEST_ASSERT(fused.metadata.ok);
  TEST_ASSERT(fused.metadata.read_count == 1u);
  TEST_ASSERT(fused.metadata.write_count == 1u);
  TEST_ASSERT(fused.metadata.param_storage.size() == 8u);

  const rund::kernel::ComputeFusedMapChainIR public_fused =
      rund::kernel::BuildFusedComputeMapChainIR(
          pair.chain, 2u, pair.graph, pair.policy,
          rund::kernel::ComputeApi::Metal);
  TEST_ASSERT(public_fused.ok);
  TEST_ASSERT(public_fused.ir.op_hash_hi == fused.ir.op_hash_hi);
  TEST_ASSERT(public_fused.ir.op_hash_lo == fused.ir.op_hash_lo);
  TEST_ASSERT(public_fused.ir.canonical_bytes == fused.ir.canonical_bytes);
  const rund::kernel::LoweringArtifact wrapper_artifact =
      rund::kernel::LowerComputeIR(fused.ir, rund::kernel::ComputeApi::Metal);

  rund::kernel::ComputeIR rejected_ir = public_fused.ir;
  rund::kernel::ExecutionMetadata rejected_metadata = public_fused.metadata;
  const auto rejected_emission = rund::kernel::compute_lowering_detail::
      EmitGeneratedRetainedComputeArtifact(std::move(rejected_ir),
                                           std::move(rejected_metadata), {});
  TEST_ASSERT(!rejected_emission.artifact.ok);
  TEST_ASSERT(rejected_emission.emission_count == 0u);
  TEST_ASSERT(std::string_view{rejected_emission.artifact.reason} ==
              "compute_lowering_invalid");
  TEST_ASSERT(rejected_ir.canonical_bytes.empty());
  TEST_ASSERT(rejected_ir.canonical_bytes.capacity() == 0u);

  rund::kernel::ComputeIR metadata_rejected_ir = public_fused.ir;
  auto metadata_input =
      rund::kernel::compute_lowering_detail::AdmitComputeInput(
          metadata_rejected_ir, rund::kernel::ComputeApi::Metal);
  const auto metadata_rejected_emission = rund::kernel::
      compute_lowering_detail::EmitGeneratedRetainedComputeArtifact(
          std::move(metadata_rejected_ir), rund::kernel::ExecutionMetadata{},
          std::move(metadata_input));
  TEST_ASSERT(!metadata_rejected_emission.artifact.ok);
  TEST_ASSERT(metadata_rejected_emission.emission_count == 1u);
  TEST_ASSERT(std::string_view{metadata_rejected_emission.artifact.reason} ==
              "compute_ir_invalid");
  TEST_ASSERT(metadata_rejected_ir.canonical_bytes.empty());
  TEST_ASSERT(metadata_rejected_ir.canonical_bytes.capacity() == 0u);

  const rund::kernel::u8 *const param_data =
      fused.metadata.param_storage.data();
  const auto emitted = rund::kernel::compute_lowering_detail::
      EmitGeneratedRetainedComputeArtifact(std::move(fused.ir),
                                           std::move(fused.metadata),
                                           std::move(admitted.input));
  TEST_ASSERT(emitted.emission_count == 1u);
  const rund::kernel::LoweringArtifact &artifact = emitted.artifact;
  TEST_ASSERT(artifact.ok);
  TEST_ASSERT(emitted.input.ok);
  TEST_ASSERT(emitted.parse_count() == 0u);
  TEST_ASSERT(fused.ir.canonical_bytes.empty());
  TEST_ASSERT(fused.ir.canonical_bytes.capacity() == 0u);
  TEST_ASSERT(artifact.canonical_ir_bytes.empty());
  TEST_ASSERT(artifact.canonical_ir_bytes.capacity() == 0u);
  TEST_ASSERT(artifact.metadata.param_storage.data() == param_data);
  TEST_ASSERT(!artifact.source_text.empty());

  TEST_ASSERT(wrapper_artifact.key == artifact.key);
  TEST_ASSERT(wrapper_artifact.kind == artifact.kind);
  TEST_ASSERT(wrapper_artifact.source_text == artifact.source_text);
  TEST_ASSERT(!wrapper_artifact.canonical_ir_bytes.empty());
  return 0;
}

} // namespace

int RunPair() {
  return test_compute_fusion_builds_checked_fused_ir_for_two_map_chain();
}

} // namespace program_compute_contract::fusion_build_contract

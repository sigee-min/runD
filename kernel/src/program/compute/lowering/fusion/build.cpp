#include <kernel/program/compute/lowering/fusion/build.hpp>

#include "internal.hpp"

#include <kernel/program/compute/lowering/admission.hpp>
#include <kernel/program/compute/lowering/fusion/graph.hpp>
#include <kernel/program/compute/lowering/fusion/result.hpp>
#include <kernel/program/compute/lowering/metadata.hpp>
#include <kernel/program/compute/lowering/resource.hpp>

#include <utility>
#include <vector>

namespace rund::kernel {
namespace compute_lowering_detail {
namespace {

[[nodiscard]] AdmittedFusedMapChainIR
RejectAdmittedFusedMapChain(const FusionPlan &fusion, const char *const reason,
                            const u32 source_parse_count = 0u) {
  return AdmittedFusedMapChainIR{
      .value = RejectFusedMapChain(fusion, reason),
      .source_parse_count = source_parse_count,
  };
}

template <class Source, class Admission>
[[nodiscard]] AdmittedFusedMapChainIR BuildAdmittedFusedComputeMapChainIRFrom(
    const Source &source, const Admission &admission, const u64 chain_count,
    const Graph &graph, const FusionPolicy &policy, const ComputeApi api) {
  const FusionPlan fusion = PlanFusion(graph, policy);
  if (!fusion.ok) {
    return RejectAdmittedFusedMapChain(fusion, fusion.reason);
  }
  if (!ComputeApiValid(api)) {
    return RejectAdmittedFusedMapChain(fusion, "compute_api_unsupported");
  }
  if (chain_count < 2u || graph.nodes == nullptr ||
      graph.node_count != chain_count || fusion.fused_node_count != 1u ||
      fusion.rejected_edge_count != 0u || !ComputeScalarValid(graph.scalar)) {
    return RejectAdmittedFusedMapChain(fusion,
                                       "compute_fusion_dependency_conflict");
  }

  std::vector<FusedSource> sources{};
  sources.reserve(static_cast<std::size_t>(chain_count));
  u64 binding_total = 0u;
  u64 node_total = 0u;
  u32 source_parse_count = 0u;
  for (u64 index = 0u; index < chain_count; ++index) {
    const ComputeIR *const ir = source(index);
    const GraphNode &node = graph.nodes[index];
    if (ir == nullptr || node.kind != NodeKind::Map ||
        ir->scalar != graph.scalar || ir->domain != graph.domain ||
        (graph.domain == ComputeDomain::Fixed &&
         ir->fixed_format != graph.fixed_format)) {
      return RejectAdmittedFusedMapChain(
          fusion, "compute_fusion_dependency_conflict", source_parse_count);
    }
    if (node.op_hash_hi != ir->op_hash_hi ||
        node.op_hash_lo != ir->op_hash_lo) {
      return RejectAdmittedFusedMapChain(fusion, "compute_ir_hash_mismatch",
                                         source_parse_count);
    }
    ComputeInputAdmission *const retained = admission(index);
    ComputeInputAdmission admitted = retained == nullptr
                                         ? AdmitComputeInput(*ir, api)
                                         : std::move(*retained);
    source_parse_count += admitted.parse_count;
    if (!admitted.ok) {
      return RejectAdmittedFusedMapChain(fusion, admitted.reason,
                                         source_parse_count);
    }
    if (admitted.key.api != api || admitted.key.scalar != ir->scalar ||
        admitted.key.domain != ir->domain ||
        admitted.key.fixed_format != ir->fixed_format ||
        admitted.key.op_hash_hi != ir->op_hash_hi ||
        admitted.key.op_hash_lo != ir->op_hash_lo ||
        admitted.key.canonical_ir_hash_hi != ir->op_hash_hi ||
        admitted.key.canonical_ir_hash_lo != ir->op_hash_lo) {
      return RejectAdmittedFusedMapChain(fusion, "compute_ir_hash_mismatch",
                                         source_parse_count);
    }
    const u32 write_bindings =
        CountBindingsOfKind(admitted.parsed, kWriteBindingKind);
    const u32 graph_writes = CountGraphBuffersOfRole(node, BufferRole::Write);
    const u32 write_nodes = WriteNodeCount(admitted.parsed);
    const bool final_source = index + 1u == chain_count;
    if (write_bindings == 0u || write_bindings != graph_writes ||
        write_bindings != write_nodes ||
        (!final_source && write_bindings != 1u)) {
      return RejectAdmittedFusedMapChain(
          fusion, "compute_fusion_dependency_conflict", source_parse_count);
    }
    const FusionNodePolicy &node_policy = policy.nodes[index];
    if (!node_policy.supported ||
        node_policy.binding_count != admitted.parsed.bindings.size() ||
        node_policy.ir_node_count != admitted.parsed.nodes.size()) {
      return RejectAdmittedFusedMapChain(
          fusion, "compute_fusion_policy_invalid", source_parse_count);
    }

    u32 read_binding = kNoBinding;
    if (index != 0u) {
      const u64 intermediate = IntermediateLogicalId(graph, index - 1u);
      const u32 read_ordinal = GraphReadOrdinalForLogicalId(node, intermediate);
      read_binding = BindingIndexForOrdinal(admitted.parsed, kReadBindingKind,
                                            read_ordinal);
      if (intermediate == 0u ||
          read_binding >= admitted.parsed.bindings.size() ||
          ReadNodeCount(admitted.parsed, read_binding) != 1u) {
        return RejectAdmittedFusedMapChain(
            fusion, "compute_fusion_dependency_conflict", source_parse_count);
      }
    }
    binding_total += admitted.parsed.bindings.size();
    node_total += admitted.parsed.nodes.size();
    sources.push_back(FusedSource{
        .parsed = std::move(admitted.parsed),
        .intermediate_read_binding = read_binding,
    });
  }

  const u64 removed = 2u * (chain_count - 1u);
  if (binding_total <= removed || node_total <= removed) {
    return RejectAdmittedFusedMapChain(
        fusion, "compute_fusion_dependency_conflict", source_parse_count);
  }
  const u64 fused_binding_count = binding_total - removed;
  const u64 fused_node_count = node_total - removed;
  if (fused_binding_count > kMaxComputeBindingCount ||
      fused_node_count > kMaxComputeNodeCount) {
    return RejectAdmittedFusedMapChain(
        fusion, "compute_fusion_capacity_boundary", source_parse_count);
  }
  BuiltFusedParsed built =
      BuildFusedParsed(sources, graph.scalar, graph.domain,
                       static_cast<u32>(fused_binding_count),
                       static_cast<u32>(fused_node_count));
  if (!built.ok) {
    return RejectAdmittedFusedMapChain(fusion, built.reason,
                                       source_parse_count);
  }

  ComputeIR fused_ir = BuildFusedIR(built.parsed, graph.scalar, graph.domain);
  ComputeInputAdmission fused_input =
      AdmitGeneratedComputeInput(fused_ir, api, std::move(built.parsed));
  if (!fused_input.ok) {
    return RejectAdmittedFusedMapChain(fusion, fused_input.reason,
                                       source_parse_count);
  }
  ExecutionMetadata metadata =
      MetadataFromParsed(fused_ir, api, fused_input.parsed);
  if (!metadata.ok) {
    return RejectAdmittedFusedMapChain(fusion, metadata.reason,
                                       source_parse_count);
  }
  return AdmittedFusedMapChainIR{
      .value = ComputeFusedMapChainIR{
          .ir = std::move(fused_ir),
          .metadata = std::move(metadata),
          .fusion = fusion,
          .ok = true,
          .reason = "ok",
      },
      .input = std::move(fused_input),
      .source_parse_count = source_parse_count,
  };
}

} // namespace

AdmittedFusedMapChainIR BuildAdmittedFusedComputeMapChainIR(
    const ComputeIR *const chain, const u64 chain_count, const Graph &graph,
    const FusionPolicy &policy, const ComputeApi api) {
  return BuildAdmittedFusedComputeMapChainIRFrom(
      [chain](const u64 index) -> const ComputeIR * {
        return chain == nullptr ? nullptr : chain + index;
      },
      [](const u64) -> ComputeInputAdmission * { return nullptr; }, chain_count,
      graph, policy, api);
}

AdmittedFusedMapChainIR BuildAdmittedFusedComputeMapChainIR(
    const ComputeIR *const *const chain, const u64 chain_count,
    const Graph &graph, const FusionPolicy &policy, const ComputeApi api) {
  return BuildAdmittedFusedComputeMapChainIRFrom(
      [chain](const u64 index) -> const ComputeIR * {
        return chain == nullptr ? nullptr : chain[index];
      },
      [](const u64) -> ComputeInputAdmission * { return nullptr; }, chain_count,
      graph, policy, api);
}

AdmittedFusedMapChainIR BuildAdmittedFusedComputeMapChainIR(
    const ComputeIR *const *const chain,
    ComputeInputAdmission *const *const inputs, const u64 chain_count,
    const Graph &graph, const FusionPolicy &policy, const ComputeApi api) {
  return BuildAdmittedFusedComputeMapChainIRFrom(
      [chain](const u64 index) -> const ComputeIR * {
        return chain == nullptr ? nullptr : chain[index];
      },
      [inputs](const u64 index) -> ComputeInputAdmission * {
        return inputs == nullptr ? nullptr : inputs[index];
      },
      chain_count, graph, policy, api);
}

} // namespace compute_lowering_detail

ComputeFusedMapChainIR
BuildFusedComputeMapChainIR(const ComputeIR *const chain,
                            const u64 chain_count, const Graph &graph,
                            const FusionPolicy &policy, const ComputeApi api) {
  compute_lowering_detail::AdmittedFusedMapChainIR admitted =
      compute_lowering_detail::BuildAdmittedFusedComputeMapChainIR(
          chain, chain_count, graph, policy, api);
  return std::move(admitted.value);
}

} // namespace rund::kernel

#include <kernel/program/compute/lowering/fusion/build.hpp>

#include <kernel/program/compute/lowering/admission.hpp>
#include <kernel/program/compute/lowering/fusion/graph.hpp>
#include <kernel/program/compute/lowering/fusion/result.hpp>
#include <kernel/program/compute/lowering/metadata.hpp>
#include <kernel/program/compute/lowering/serialize.hpp>

#include <algorithm>
#include <array>
#include <limits>
#include <string_view>
#include <utility>
#include <vector>

namespace rund::kernel {
namespace compute_lowering_detail {

inline constexpr u32 kNoBinding = ~u32{0u};

struct FusedSource {
  ParsedIR parsed{};
  u32 intermediate_read_binding = kNoBinding;
};

[[nodiscard]] u32 ReadNodeCount(const ParsedIR &parsed,
                                const u32 binding_index) noexcept {
  u32 count = 0u;
  for (const ParsedNode &node : parsed.nodes) {
    count += static_cast<u32>(node.op == static_cast<u8>(IrOp::Read) &&
                              node.aux == binding_index);
  }
  return count;
}

inline ParsedBinding PrefixedBinding(const u64 source_index,
                                     const ParsedBinding &binding) {
  ParsedBinding out = binding;
  out.name = "f";
  out.name += std::to_string(source_index);
  out.name += "_";
  out.name += binding.name;
  return out;
}

struct FusedBindingMap {
  std::vector<ParsedBinding> bindings{};
  std::vector<u32> indices{};
  std::vector<u32> offsets{};
};

[[nodiscard]] inline std::size_t
BindingMapIndex(const FusedBindingMap &map, const std::size_t source_index,
                const std::size_t binding_index) noexcept {
  return static_cast<std::size_t>(map.offsets[source_index]) + binding_index;
}

[[nodiscard]] inline FusedBindingMap
BuildFusedBindingMap(const std::vector<FusedSource> &sources,
                     const u32 binding_count) {
  FusedBindingMap map{};
  map.bindings.reserve(binding_count);
  map.offsets.resize(sources.size() + 1u);
  u32 source_binding_count = 0u;
  for (std::size_t source_index = 0u; source_index < sources.size();
       ++source_index) {
    map.offsets[source_index] = source_binding_count;
    source_binding_count +=
        static_cast<u32>(sources[source_index].parsed.bindings.size());
  }
  map.offsets[sources.size()] = source_binding_count;
  map.indices.resize(source_binding_count);
  constexpr std::array<u8, 3u> kinds{
      kParamBindingKind,
      kReadBindingKind,
      kWriteBindingKind,
  };
  for (const u8 kind : kinds) {
    for (std::size_t source_index = 0u; source_index < sources.size();
         ++source_index) {
      const FusedSource &source = sources[source_index];
      for (std::size_t binding_index = 0u;
           binding_index < source.parsed.bindings.size(); ++binding_index) {
        const ParsedBinding &binding = source.parsed.bindings[binding_index];
        if (binding.kind != kind ||
            (kind == kReadBindingKind && source_index != 0u &&
             binding_index == source.intermediate_read_binding) ||
            (kind == kWriteBindingKind &&
             source_index + 1u != sources.size())) {
          continue;
        }
        map.indices[BindingMapIndex(map, source_index, binding_index)] =
            static_cast<u32>(map.bindings.size());
        map.bindings.push_back(PrefixedBinding(source_index, binding));
      }
    }
  }
  return map;
}

inline void RemapFusedValueNode(
    ParsedNode &node, const IrOp op,
    const std::array<u32, kMaxComputeNodeCount + 1u> &node_map) {
  if (op == IrOp::Constant) {
    return;
  }
  if (UnaryValueOp(op) || ConstShiftOp(op)) {
    node.lhs = node_map[node.lhs];
  } else if (BinaryValueOp(op)) {
    node.lhs = node_map[node.lhs];
    node.rhs = node_map[node.rhs];
  } else if (TernaryValueOp(op)) {
    node.lhs = node_map[node.lhs];
    node.rhs = node_map[node.rhs];
    node.aux = node_map[node.aux];
  }
}

struct FusedNodeMap final {
  std::vector<ParsedNode> nodes{};
  bool ok = false;
};

[[nodiscard]] inline FusedNodeMap
BuildFusedNodeMap(const std::vector<FusedSource> &sources,
                  const FusedBindingMap &bindings, const u32 node_count) {
  FusedNodeMap out{};
  out.nodes.reserve(node_count);
  std::array<u32, kMaxComputeNodeCount + 1u> node_map{};
  u32 carrier = 0u;
  for (std::size_t source_index = 0u; source_index < sources.size();
       ++source_index) {
    const FusedSource &source = sources[source_index];
    std::fill(node_map.begin(),
              node_map.begin() + source.parsed.nodes.size() + 1u, 0u);
    for (std::size_t node_index = 0u; node_index < source.parsed.nodes.size();
         ++node_index) {
      ParsedNode node = source.parsed.nodes[node_index];
      const u32 current = static_cast<u32>(node_index + 1u);
      const IrOp op = static_cast<IrOp>(node.op);
      if (source_index != 0u && op == IrOp::Read &&
          node.aux == source.intermediate_read_binding) {
        if (carrier == 0u) {
          return out;
        }
        node_map[current] = carrier;
        continue;
      }
      if (op == IrOp::Write && source_index + 1u != sources.size()) {
        carrier = node_map[node.lhs];
        if (carrier == 0u) {
          return out;
        }
        continue;
      }
      if (op == IrOp::Param || op == IrOp::Read ||
          op == IrOp::ReadUniform) {
        node.aux =
            bindings.indices[BindingMapIndex(bindings, source_index, node.aux)];
      } else if (op == IrOp::ReadAt) {
        node.lhs =
            bindings.indices[BindingMapIndex(bindings, source_index, node.lhs)];
        node.aux =
            bindings.indices[BindingMapIndex(bindings, source_index, node.aux)];
      } else if (op == IrOp::Write) {
        if (node.lhs != 0u) {
          node.lhs = node_map[node.lhs];
        }
        node.aux =
            bindings.indices[BindingMapIndex(bindings, source_index, node.aux)];
      } else {
        RemapFusedValueNode(node, op, node_map);
      }
      out.nodes.push_back(node);
      node_map[current] = static_cast<u32>(out.nodes.size());
    }
  }
  out.ok = out.nodes.size() == node_count;
  return out;
}

[[nodiscard]] inline std::string
FusedIrName(const std::vector<FusedSource> &sources) {
  std::size_t capacity = 6u;
  for (const FusedSource &source : sources) {
    capacity += source.parsed.name.size() + 1u;
  }
  std::string name = "fused.";
  name.reserve(capacity);
  for (std::size_t index = 0u; index < sources.size(); ++index) {
    if (index != 0u) {
      name += ".";
    }
    name += sources[index].parsed.name;
  }
  return name;
}

[[nodiscard]] inline std::vector<u8>
BuildFusedCanonicalBytes(const ParsedIR &parsed) {
  std::vector<u8> bytes{};
  std::size_t capacity = 4u + std::string_view{"rund.compute.ir"}.size() + 4u +
                         parsed.name.size() + 6u + 4u + 4u;
  bool capacity_ok = true;
  const auto add = [&capacity, &capacity_ok](const std::size_t value) {
    if (!capacity_ok ||
        value > std::numeric_limits<std::size_t>::max() - capacity) {
      capacity_ok = false;
      return;
    }
    capacity += value;
  };
  for (const ParsedBinding &binding : parsed.bindings) {
    add(kMinBindingBytes);
    add(binding.name.size());
    add(binding.value_bytes.size());
  }
  if (parsed.nodes.size() >
      std::numeric_limits<std::size_t>::max() / kSerializedNodeBytes) {
    capacity_ok = false;
  } else {
    add(parsed.nodes.size() * kSerializedNodeBytes);
  }
  if (capacity_ok) {
    bytes.reserve(capacity);
  }
  AppendSerializedBytes(bytes, "rund.compute.ir");
  AppendSerializedBytes(bytes, parsed.name);
  AppendSerializedU8(bytes, parsed.scalar_mode);
  AppendSerializedU8(bytes, parsed.fixed_format.integer_bits);
  AppendSerializedU8(bytes, parsed.fixed_format.fraction_bits);
  AppendSerializedU8(bytes, static_cast<u8>(parsed.fixed_format.rounding));
  AppendSerializedU8(bytes, static_cast<u8>(parsed.fixed_format.overflow));
  AppendSerializedU8(bytes, static_cast<u8>(parsed.fixed_format.approximation));
  AppendSerializedU32(bytes, static_cast<u32>(parsed.bindings.size()));
  for (const ParsedBinding &binding : parsed.bindings) {
    AppendSerializedBinding(bytes, binding);
  }
  AppendSerializedU32(bytes, static_cast<u32>(parsed.nodes.size()));
  for (const ParsedNode &node : parsed.nodes) {
    AppendSerializedNode(bytes, node);
  }
  return bytes;
}

struct BuiltFusedParsed {
  ParsedIR parsed{};
  bool ok = false;
  const char *reason = "compute_fusion_invalid";
};

[[nodiscard]] inline BuiltFusedParsed
BuildFusedParsed(const std::vector<FusedSource> &sources,
                 const ComputeScalar scalar, const ComputeDomain domain,
                 const u32 binding_count, const u32 node_count) {
  FusedBindingMap bindings = BuildFusedBindingMap(sources, binding_count);
  if (bindings.bindings.size() != binding_count ||
      bindings.bindings.size() > kMaxComputeBindingCount) {
    return BuiltFusedParsed{.reason = "compute_ir_binding_count_invalid"};
  }
  FusedNodeMap nodes = BuildFusedNodeMap(sources, bindings, node_count);
  if (!nodes.ok || nodes.nodes.empty() ||
      nodes.nodes.size() > kMaxComputeNodeCount) {
    return BuiltFusedParsed{.reason = "compute_ir_node_count_invalid"};
  }
  ParsedIR parsed{
      .name = FusedIrName(sources),
      .scalar_mode = DomainModeFor(scalar, domain),
      .fixed_format = sources.back().parsed.fixed_format,
      .bindings = std::move(bindings.bindings),
      .nodes = std::move(nodes.nodes),
      .ok = true,
      .reason = "ok",
  };
  return BuiltFusedParsed{
      .parsed = std::move(parsed),
      .ok = true,
      .reason = "ok",
  };
}

[[nodiscard]] inline ComputeIR BuildFusedIR(ParsedIR &parsed,
                                            const ComputeScalar scalar,
                                            const ComputeDomain domain) {
  std::vector<u8> bytes = BuildFusedCanonicalBytes(parsed);
  const compute_ir_detail::ComputeIrHash hash =
      compute_ir_detail::HashComputeIrCanonicalBytes(
          bytes.empty() ? nullptr : bytes.data(),
          static_cast<u64>(bytes.size()));
  return ComputeIR{
      .scalar = scalar,
      .domain = domain,
      .fixed_format = parsed.fixed_format,
      .op_hash_hi = hash.hi,
      .op_hash_lo = hash.lo,
      .canonical_bytes = std::move(bytes),
      .ok = true,
      .reason = "ok",
  };
}

[[nodiscard]] inline AdmittedFusedMapChainIR
RejectAdmittedFusedMapChain(const FusionPlan &fusion, const char *const reason,
                            const u32 source_parse_count = 0u) {
  return AdmittedFusedMapChainIR{
      .value = RejectFusedMapChain(fusion, reason),
      .source_parse_count = source_parse_count,
  };
}

template <class Source, class Admission>
[[nodiscard]] inline AdmittedFusedMapChainIR
BuildAdmittedFusedComputeMapChainIRFrom(
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
      fusion.rejected_edge_count != 0u) {
    return RejectAdmittedFusedMapChain(fusion,
                                       "compute_fusion_dependency_conflict");
  }
  if (!ComputeScalarValid(graph.scalar)) {
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
      .value =
          ComputeFusedMapChainIR{
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

[[nodiscard]] AdmittedFusedMapChainIR BuildAdmittedFusedComputeMapChainIR(
    const ComputeIR *const chain, const u64 chain_count, const Graph &graph,
    const FusionPolicy &policy, const ComputeApi api) {
  return BuildAdmittedFusedComputeMapChainIRFrom(
      [chain](const u64 index) -> const ComputeIR * {
        return chain == nullptr ? nullptr : chain + index;
      },
      [](const u64) -> ComputeInputAdmission * { return nullptr; }, chain_count,
      graph, policy, api);
}

[[nodiscard]] AdmittedFusedMapChainIR BuildAdmittedFusedComputeMapChainIR(
    const ComputeIR *const *const chain, const u64 chain_count,
    const Graph &graph, const FusionPolicy &policy, const ComputeApi api) {
  return BuildAdmittedFusedComputeMapChainIRFrom(
      [chain](const u64 index) -> const ComputeIR * {
        return chain == nullptr ? nullptr : chain[index];
      },
      [](const u64) -> ComputeInputAdmission * { return nullptr; }, chain_count,
      graph, policy, api);
}

[[nodiscard]] AdmittedFusedMapChainIR BuildAdmittedFusedComputeMapChainIR(
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

[[nodiscard]] ComputeFusedMapChainIR
BuildFusedComputeMapChainIR(const ComputeIR *const chain, const u64 chain_count,
                            const Graph &graph, const FusionPolicy &policy,
                            const ComputeApi api) {
  compute_lowering_detail::AdmittedFusedMapChainIR admitted =
      compute_lowering_detail::BuildAdmittedFusedComputeMapChainIR(
          chain, chain_count, graph, policy, api);
  return std::move(admitted.value);
}

} // namespace rund::kernel

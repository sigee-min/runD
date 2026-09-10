#include "internal.hpp"

#include <kernel/program/compute/lowering/metadata.hpp>
#include <kernel/program/compute/lowering/resource.hpp>
#include <kernel/program/compute/lowering/serialize.hpp>

#include <algorithm>
#include <array>
#include <limits>
#include <string_view>
#include <utility>

namespace rund::kernel::compute_lowering_detail {
namespace {

ParsedBinding PrefixedBinding(const u64 source_index,
                              const ParsedBinding &binding) {
  ParsedBinding out = binding;
  out.name = "f";
  out.name += std::to_string(source_index);
  out.name += "_";
  out.name += binding.name;
  return out;
}

[[nodiscard]] std::size_t
BindingMapIndex(const FusedBindingMap &map, const std::size_t source_index,
                const std::size_t binding_index) noexcept {
  return static_cast<std::size_t>(map.offsets[source_index]) + binding_index;
}

[[nodiscard]] bool RemapFusedValueNode(
    ParsedNode &node,
    const std::array<u32, kMaxComputeNodeCount + 1u> &node_map) noexcept {
  const ParsedNodeResources resources = ParsedNodeResourcesFor(node);
  if (!resources.ok || !resources.produces_value) {
    return false;
  }
  std::array<u32 *, 3u> operands{&node.lhs, &node.rhs, &node.aux};
  for (u32 index = 0u; index < resources.ref_count; ++index) {
    const u32 source = resources.refs[index];
    if (source == 0u || source >= node_map.size() ||
        *operands[index] != source || node_map[source] == 0u) {
      return false;
    }
    *operands[index] = node_map[source];
  }
  return true;
}

[[nodiscard]] std::string FusedIrName(
    const std::vector<FusedSource> &sources) {
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

[[nodiscard]] std::vector<u8>
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

} // namespace

u32 ReadNodeCount(const ParsedIR &parsed, const u32 binding_index) noexcept {
  u32 count = 0u;
  for (const ParsedNode &node : parsed.nodes) {
    count += static_cast<u32>(node.op == static_cast<u8>(IrOp::Read) &&
                              node.aux == binding_index);
  }
  return count;
}

FusedBindingMap BuildFusedBindingMap(const std::vector<FusedSource> &sources,
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

FusedNodeMap BuildFusedNodeMap(const std::vector<FusedSource> &sources,
                               const FusedBindingMap &bindings,
                               const u32 node_count) {
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
      if (op == IrOp::Param || op == IrOp::Read || op == IrOp::ReadUniform) {
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
      } else if (!RemapFusedValueNode(node, node_map)) {
        return out;
      }
      out.nodes.push_back(node);
      node_map[current] = static_cast<u32>(out.nodes.size());
    }
  }
  out.ok = out.nodes.size() == node_count;
  return out;
}

BuiltFusedParsed BuildFusedParsed(const std::vector<FusedSource> &sources,
                                  const ComputeScalar scalar,
                                  const ComputeDomain domain,
                                  const u32 binding_count,
                                  const u32 node_count) {
  FusedBindingMap bindings = BuildFusedBindingMap(sources, binding_count);
  if (bindings.bindings.size() != binding_count ||
      bindings.bindings.size() > kMaxComputeBindingCount) {
    return BuiltFusedParsed{.reason = "compute_ir_binding_count_invalid"};
  }
  FusedNodeMap nodes = BuildFusedNodeMap(sources, bindings, node_count);
  if (!nodes.ok || nodes.nodes.empty() || nodes.nodes.size() > kMaxComputeNodeCount) {
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

ComputeIR BuildFusedIR(ParsedIR &parsed, const ComputeScalar scalar,
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

} // namespace rund::kernel::compute_lowering_detail

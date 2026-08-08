#include <kernel/program/compute/graph/signature.hpp>

#include <kernel/program/compute/lowering/admission.hpp>
#include <kernel/program/compute/lowering/metadata.hpp>

#include <array>
#include <cstddef>

namespace rund::kernel {
namespace compute_lowering_detail {
namespace {

[[nodiscard]] constexpr bool DomainValid(const ComputeDomain domain) noexcept {
  return domain == ComputeDomain::I32 || domain == ComputeDomain::U32 ||
         domain == ComputeDomain::I64 || domain == ComputeDomain::U64 ||
         domain == ComputeDomain::Fixed;
}

[[nodiscard]] ComputeResourceSummary RejectAnalyzedSummary() noexcept {
  return {};
}

} // namespace

ComputeResourceSummary
AnalyzeComputeResources(const ParsedIR &parsed, const ComputeScalar scalar,
                        const ComputeDomain domain) noexcept {
  if (!parsed.ok || !ComputeScalarValid(scalar) || !DomainValid(domain) ||
      parsed.scalar_mode != DomainModeFor(scalar, domain) ||
      parsed.nodes.empty() || parsed.nodes.size() > kMaxComputeNodeCount) {
    return RejectAnalyzedSummary();
  }

  std::array<u32, kMaxComputeNodeCount + 1u> last_use{};
  std::array<bool, kMaxComputeNodeCount + 1u> produces_value{};
  ComputeResourceSummary out{};
  for (std::size_t index = 0u; index < parsed.nodes.size(); ++index) {
    const ParsedNode &node = parsed.nodes[index];
    const u32 current = static_cast<u32>(index + 1u);
    const ParsedNodeResources resources = ParsedNodeResourcesFor(node);
    if (!resources.ok) {
      return RejectAnalyzedSummary();
    }
    for (u32 operand = 0u; operand < resources.ref_count; ++operand) {
      const u32 ref = resources.refs[operand];
      if (ref == 0u || ref >= current || !produces_value[ref]) {
        return RejectAnalyzedSummary();
      }
      last_use[ref] = current;
    }
    produces_value[current] = resources.produces_value;
    if (resources.produces_value) {
      // A dead structural definition is live at its defining node.
      last_use[current] = current;
    }

    switch (static_cast<IrOp>(node.op)) {
    case IrOp::Read:
      ++out.direct_read_count;
      break;
    case IrOp::ReadUniform:
      ++out.uniform_read_count;
      break;
    case IrOp::ReadAt:
      ++out.indexed_read_count;
      break;
    case IrOp::Write:
      ++out.write_count;
      break;
    default:
      break;
    }
  }
  if (out.write_count == 0u) {
    return RejectAnalyzedSummary();
  }

  const u32 words_per_value = domain == ComputeDomain::Fixed
                                  ? 4u
                                  : (scalar == ComputeScalar::Lane64 ? 2u : 1u);
  std::array<u32, kMaxComputeNodeCount + 1u> release_words{};
  for (u32 node = 1u; node <= parsed.nodes.size(); ++node) {
    if (produces_value[node]) {
      release_words[last_use[node]] += words_per_value;
    }
  }

  u32 live_words = 0u;
  for (u32 node = 1u; node <= parsed.nodes.size(); ++node) {
    if (produces_value[node]) {
      live_words += words_per_value;
    }
    if (live_words > out.peak_live_words) {
      out.peak_live_words = live_words;
    }
    if (release_words[node] > live_words) {
      return RejectAnalyzedSummary();
    }
    live_words -= release_words[node];
  }
  if (live_words != 0u || out.peak_live_words == 0u) {
    return RejectAnalyzedSummary();
  }

  out.analysis_version = kComputeResourceAnalysisVersion;
  out.ok = true;
  out.reason = "ok";
  return out;
}

} // namespace compute_lowering_detail
namespace {

[[nodiscard]] ExecutionMetadata RejectMetadata(const char *const reason) {
  return ExecutionMetadata{.reason = reason};
}

[[nodiscard]] ComputeResourceSummary
RejectResourceSummary(const char *const reason) noexcept {
  return ComputeResourceSummary{.reason = reason};
}

[[nodiscard]] GraphSignature RejectMapSignature(const char *const reason) {
  return GraphSignature{.kind = NodeKind::Map, .reason = reason};
}

[[nodiscard]] GraphSignature GraphSignatureFromParsedMap(
    const compute_lowering_detail::ParsedIR &parsed) noexcept {
  u64 read_count = 0u;
  u64 write_count = 0u;
  u64 edge_count = 0u;
  for (const compute_lowering_detail::ParsedBinding &binding :
       parsed.bindings) {
    if (binding.kind == compute_lowering_detail::kReadBindingKind) {
      ++read_count;
      ++edge_count;
    } else if (binding.kind == compute_lowering_detail::kWriteBindingKind) {
      ++write_count;
      ++edge_count;
    }
  }

  const bool metadata_ok = write_count != 0u;
  const char *const metadata_reason =
      metadata_ok ? "ok" : "compute_ir_node_invalid";
  if (!metadata_ok || read_count > kMaxGraphSignatureValues ||
      edge_count > kMaxGraphSignatureValues) {
    return RejectMapSignature(metadata_reason);
  }

  GraphSignature out{.kind = NodeKind::Map, .ok = true, .reason = "ok"};
  for (const compute_lowering_detail::ParsedBinding &binding :
       parsed.bindings) {
    if (binding.kind != compute_lowering_detail::kReadBindingKind &&
        binding.kind != compute_lowering_detail::kWriteBindingKind) {
      continue;
    }
    const bool write =
        binding.kind == compute_lowering_detail::kWriteBindingKind;
    out.values[out.value_count] = GraphValueType{
        .kind = write ? GraphValueKind::Output : GraphValueKind::Values,
        .role = write ? BufferRole::Write : BufferRole::Read,
        .element_bytes = binding.element_bytes,
        .count = 0u,
    };
    ++out.value_count;
    if (write) {
      ++out.output_count;
    }
  }
  return out;
}

} // namespace

u64 RequiredInputCount(const ExecutionMetadata &metadata, const u64 binding,
                       const u64 tile_count) noexcept {
  if (binding >= 64u || binding >= metadata.read_count || tile_count == 0u) {
    return 0u;
  }
  const u64 bit = u64{1u} << binding;
  u64 required = (metadata.direct_read_mask & bit) != 0u    ? tile_count
                 : (metadata.uniform_read_mask & bit) != 0u ? 1u
                                                            : 0u;
  for (const ReadRoute route : metadata.read_routes) {
    if (route.index == binding) {
      required = tile_count;
    } else if (route.source == binding && route.count > required) {
      required = route.count;
    }
  }
  return required;
}

ExecutionMetadata BuildExecutionMetadata(const ComputeIR &ir,
                                         const ComputeApi api) {
  const compute_lowering_detail::ComputeInputAdmission input =
      compute_lowering_detail::AdmitComputeInput(ir, api);
  if (!input.ok) {
    return RejectMetadata(input.reason);
  }

  return compute_lowering_detail::MetadataFromParsed(ir, api, input.parsed);
}

ComputeResourceSummary BuildComputeResourceSummary(const ComputeIR &ir,
                                                   const ComputeApi api) {
  const compute_lowering_detail::ComputeInputAdmission input =
      compute_lowering_detail::AdmitComputeInput(ir, api);
  if (!input.ok) {
    return RejectResourceSummary(input.reason);
  }
  return compute_lowering_detail::AnalyzeComputeResources(input.parsed,
                                                          ir.scalar, ir.domain);
}

GraphSignature BuildMapGraphSignature(const ComputeIR &ir,
                                      const ComputeApi api) {
  const compute_lowering_detail::ComputeInputAdmission input =
      compute_lowering_detail::AdmitComputeInput(ir, api);
  if (!input.ok) {
    return RejectMapSignature(input.reason);
  }
  return GraphSignatureFromParsedMap(input.parsed);
}

} // namespace rund::kernel

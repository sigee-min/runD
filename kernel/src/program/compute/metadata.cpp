#include <kernel/program/compute/graph/signature.hpp>

#include <kernel/program/compute/lowering/admission.hpp>
#include <kernel/program/compute/lowering/metadata.hpp>

namespace rund::kernel {
namespace {

[[nodiscard]] ExecutionMetadata RejectMetadata(
    const char* const reason) {
  return ExecutionMetadata{.reason = reason};
}

[[nodiscard]] GraphSignature RejectMapSignature(const char* const reason) {
  return GraphSignature{.kind = NodeKind::Map, .reason = reason};
}

[[nodiscard]] GraphSignature GraphSignatureFromParsedMap(
    const compute_lowering_detail::ParsedIR& parsed) noexcept {
  u64 read_count = 0u;
  u64 write_count = 0u;
  u64 edge_count = 0u;
  for (const compute_lowering_detail::ParsedBinding& binding :
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
  const char* const metadata_reason =
      metadata_ok ? "ok" : "compute_ir_node_invalid";
  if (!metadata_ok || read_count > kMaxGraphSignatureValues ||
      edge_count > kMaxGraphSignatureValues) {
    return RejectMapSignature(metadata_reason);
  }

  GraphSignature out{.kind = NodeKind::Map, .ok = true, .reason = "ok"};
  for (const compute_lowering_detail::ParsedBinding& binding :
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

}  // namespace

u64 RequiredInputCount(const ExecutionMetadata &metadata, const u64 binding,
                       const u64 tile_count) noexcept {
  if (binding >= 64u || binding >= metadata.read_count ||
      tile_count == 0u) {
    return 0u;
  }
  u64 required =
      (metadata.direct_read_mask & (u64{1u} << binding)) != 0u ? tile_count
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

ExecutionMetadata BuildExecutionMetadata(
    const ComputeIR& ir,
    const ComputeApi api) {
  const compute_lowering_detail::ComputeInputAdmission input =
      compute_lowering_detail::AdmitComputeInput(ir, api);
  if (!input.ok) {
    return RejectMetadata(input.reason);
  }

  return compute_lowering_detail::MetadataFromParsed(ir, api, input.parsed);
}

GraphSignature BuildMapGraphSignature(
    const ComputeIR& ir,
    const ComputeApi api) {
  const compute_lowering_detail::ComputeInputAdmission input =
      compute_lowering_detail::AdmitComputeInput(ir, api);
  if (!input.ok) {
    return RejectMapSignature(input.reason);
  }
  return GraphSignatureFromParsedMap(input.parsed);
}

}  // namespace rund::kernel

#include "local.hpp"

#include <kernel/program/compute/graph/schema.hpp>
#include <kernel/program/compute/metadata.hpp>

#include <cstddef>

namespace rund::kernel {

[[nodiscard]] GraphSignature
GraphSignatureFor(const ExecutionMetadata &metadata) noexcept {
  if (!metadata.ok ||
      metadata.binding_accesses.size() != metadata.binding_names.size() ||
      metadata.read_count > kMaxGraphSignatureValues ||
      metadata.write_count == 0u ||
      metadata.binding_accesses.size() > kMaxGraphSignatureValues ||
      metadata.input_element_bytes.size() !=
          static_cast<std::size_t>(metadata.read_count) ||
      metadata.output_element_bytes.size() !=
          static_cast<std::size_t>(metadata.write_count)) {
    return GraphSignature{.kind = NodeKind::Map, .reason = metadata.reason};
  }

  GraphSignature out{
      .kind = NodeKind::Map,
      .ok = true,
      .reason = "ok",
  };
  std::size_t read_index = 0u;
  std::size_t write_index = 0u;
  for (std::size_t index = 0u; index < metadata.binding_accesses.size();
       ++index) {
    const BufferRole role = GraphRoleFor(metadata.binding_accesses[index]);
    u64 element_bytes = 0u;
    GraphValueKind kind = GraphValueKind::Values;
    if (role == BufferRole::Read) {
      if (read_index >= metadata.input_element_bytes.size()) {
        return GraphSignature{.kind = NodeKind::Map,
                              .reason = "compute_graph_signature_invalid"};
      }
      element_bytes = metadata.input_element_bytes[read_index];
      ++read_index;
    } else if (role == BufferRole::Write) {
      if (write_index >= metadata.output_element_bytes.size()) {
        return GraphSignature{.kind = NodeKind::Map,
                              .reason = "compute_graph_signature_invalid"};
      }
      element_bytes = metadata.output_element_bytes[write_index];
      ++write_index;
      kind = GraphValueKind::Output;
    } else {
      return GraphSignature{.kind = NodeKind::Map,
                            .reason = "compute_graph_signature_invalid"};
    }
    if (element_bytes == 0u || element_bytes > static_cast<u64>(~u32{0u})) {
      return GraphSignature{.kind = NodeKind::Map,
                            .reason = "compute_graph_signature_invalid"};
    }
    out.values[out.value_count] = GraphValueType{
        .kind = kind,
        .role = role,
        .element_bytes = static_cast<u32>(element_bytes),
        .count = 0u,
    };
    ++out.value_count;
    if (role == BufferRole::Write) {
      ++out.output_count;
    }
  }
  return out;
}

} // namespace rund::kernel

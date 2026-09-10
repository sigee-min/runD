#include "local.hpp"

namespace rund::kernel::graph_signature_detail {

[[nodiscard]] GraphValueType Value(const GraphValueKind kind,
                                   const BufferRole role,
                                   const u64 element_bytes, const u64 count,
                                   const u64 rows, const u64 cols,
                                   const u64 batch_count) noexcept {
  return GraphValueType{
      .kind = kind,
      .role = role,
      .element_bytes = static_cast<u32>(element_bytes),
      .count = count,
      .rows = rows,
      .cols = cols,
      .batch_count = batch_count,
  };
}

[[nodiscard]] GraphSignature Reject(const NodeKind kind,
                                    const char *const reason) noexcept {
  return GraphSignature{.kind = kind, .reason = reason};
}

void Add(GraphSignature &signature, const GraphValueType value) noexcept {
  if (signature.value_count >= kMaxGraphSignatureValues ||
      value.element_bytes > static_cast<u64>(~u32{0u})) {
    signature.ok = false;
    signature.reason = "compute_graph_signature_invalid";
    return;
  }
  signature.values[signature.value_count] = value;
  ++signature.value_count;
  if (value.role == BufferRole::Write) {
    ++signature.output_count;
  }
  if (value.kind == GraphValueKind::Status) {
    signature.status_count = value.count;
  }
}

[[nodiscard]] GraphSignature Begin(const NodeKind kind, const bool plan_ok,
                                   const char *const plan_reason) noexcept {
  return plan_ok ? GraphSignature{.kind = kind, .ok = true, .reason = "ok"}
                 : Reject(kind, plan_reason);
}

} // namespace rund::kernel::graph_signature_detail

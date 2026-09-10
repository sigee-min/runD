#pragma once

#include <kernel/program/compute/graph/signature.hpp>

namespace rund::kernel::graph_signature_detail {

[[nodiscard]] GraphValueType Value(GraphValueKind kind, BufferRole role,
                                   u64 element_bytes, u64 count, u64 rows = 0u,
                                   u64 cols = 0u,
                                   u64 batch_count = 0u) noexcept;
[[nodiscard]] GraphSignature Reject(NodeKind kind, const char *reason) noexcept;
void Add(GraphSignature &signature, GraphValueType value) noexcept;
[[nodiscard]] GraphSignature Begin(NodeKind kind, bool plan_ok,
                                   const char *plan_reason) noexcept;

} // namespace rund::kernel::graph_signature_detail

#pragma once

#include <kernel/program/compute/limit.hpp>

namespace rund::kernel {

enum class BufferRole : u8 {
  Read = 1u,
  Write = 2u,
};

enum class BufferInit : u8 {
  Preserve = 0u,
  Zero = 1u,
};

// A graph control source is an ordinary resident read binding with an
// additional execution meaning.  It never names a host callback or a backend
// object.  Descriptor keeps the authored element count.  CountU32/CountU64
// admit the resident scalar only when it is <= capacity and use that logical
// count for the physical dispatch; an oversized count is a typed execution
// failure and dispatches no work. PredicateU32/PredicateU64 execute the node
// only when the resident scalar equals expected.  Count and predicate may be
// composed on the same node.
enum class GraphControlSource : u8 {
  Descriptor = 0u,
  U32 = 1u,
  U64 = 2u,
};

[[nodiscard]] constexpr bool
GraphControlSourceValid(const GraphControlSource source) noexcept {
  return source == GraphControlSource::Descriptor ||
         source == GraphControlSource::U32 || source == GraphControlSource::U64;
}

inline constexpr u32 kNoGraphControlBinding = ~u32{0u};

struct GraphControl final {
  GraphControlSource count_source = GraphControlSource::Descriptor;
  u32 count_binding = kNoGraphControlBinding;
  u64 count_byte_offset = 0u;
  u64 capacity = 0u;
  GraphControlSource predicate_source = GraphControlSource::Descriptor;
  u32 predicate_binding = kNoGraphControlBinding;
  u64 predicate_byte_offset = 0u;
  u64 predicate_expected = 0u;
  u32 iteration = 0u;

  [[nodiscard]] constexpr bool has_count() const noexcept {
    return count_source != GraphControlSource::Descriptor;
  }
  [[nodiscard]] constexpr bool has_predicate() const noexcept {
    return predicate_source != GraphControlSource::Descriptor;
  }
  [[nodiscard]] constexpr bool valid(const u64 buffer_count) const noexcept {
    if (!GraphControlSourceValid(count_source) ||
        !GraphControlSourceValid(predicate_source)) {
      return false;
    }
    if (has_count()) {
      if (count_binding >= buffer_count || capacity == 0u ||
          (count_source == GraphControlSource::U32 &&
           (count_byte_offset & 3u) != 0u) ||
          (count_source == GraphControlSource::U64 &&
           (count_byte_offset & 7u) != 0u)) {
        return false;
      }
    } else if (count_binding != kNoGraphControlBinding ||
               count_byte_offset != 0u) {
      return false;
    }
    if (has_predicate()) {
      if (predicate_binding >= buffer_count ||
          (predicate_source == GraphControlSource::U32 &&
           ((predicate_byte_offset & 3u) != 0u ||
            predicate_expected > ~u32{0u})) ||
          (predicate_source == GraphControlSource::U64 &&
           (predicate_byte_offset & 7u) != 0u)) {
        return false;
      }
    } else if (predicate_binding != kNoGraphControlBinding ||
               predicate_byte_offset != 0u || predicate_expected != 0u ||
               (iteration != 0u && !has_count())) {
      return false;
    }
    return has_count() || capacity == 0u;
  }
};

[[nodiscard]] constexpr ComputeBindingAccess
ComputeAccessFor(const BufferRole role) noexcept {
  return role == BufferRole::Read
             ? ComputeBindingAccess::Read
             : (role == BufferRole::Write
                    ? ComputeBindingAccess::Write
                    : static_cast<ComputeBindingAccess>(0u));
}

[[nodiscard]] constexpr BufferRole
GraphRoleFor(const ComputeBindingAccess access) noexcept {
  return access == ComputeBindingAccess::Read
             ? BufferRole::Read
             : (access == ComputeBindingAccess::Write
                    ? BufferRole::Write
                    : static_cast<BufferRole>(0u));
}

enum class NodeKind : u8 {
  Map = 1u,
  Scan = 2u,
  Sort = 3u,
  Compact = 4u,
  Gather = 8u,
  Reduce = 9u,
  Scatter = 10u,
  Partition = 11u,
  SegmentedScan = 12u,
  Stencil = 13u,
  Histogram = 14u,
  SegmentedReduce = 15u,
  Transform = 16u,
  Matrix = 17u,
  Factor = 18u,
  Solve = 19u,
  Spectrum = 20u,
  ScatterReduce = 21u,
};

[[nodiscard]] constexpr bool NodeKindValid(const NodeKind kind) noexcept {
  return kind == NodeKind::Map || kind == NodeKind::Scan ||
         kind == NodeKind::Sort || kind == NodeKind::Compact ||
         kind == NodeKind::Gather || kind == NodeKind::Reduce ||
         kind == NodeKind::Scatter || kind == NodeKind::Partition ||
         kind == NodeKind::SegmentedScan || kind == NodeKind::Stencil ||
         kind == NodeKind::Histogram || kind == NodeKind::SegmentedReduce ||
         kind == NodeKind::Transform || kind == NodeKind::Matrix ||
         kind == NodeKind::Factor || kind == NodeKind::Solve ||
         kind == NodeKind::Spectrum || kind == NodeKind::ScatterReduce;
}

enum class GraphValueKind : u8 {
  Values = 1u,
  Keys = 2u,
  Indices = 3u,
  Flags = 4u,
  Heads = 5u,
  Bins = 6u,
  Counts = 7u,
  Matrix = 8u,
  Factor = 9u,
  Aux = 10u,
  Rhs = 11u,
  Output = 12u,
  Status = 13u,
  Real = 14u,
  Imag = 15u,
  Vectors = 16u,
  LogicalCount = 17u,
};

inline constexpr u64 kMaxGraphSignatureValues = 32u;

struct GraphValueType {
  GraphValueKind kind = GraphValueKind::Values;
  BufferRole role = BufferRole::Read;
  u32 element_bytes = 0u;
  u64 count = 0u;
  u64 rows = 0u;
  u64 cols = 0u;
  u64 batch_count = 0u;
};

struct GraphSignature {
  NodeKind kind = NodeKind::Map;
  GraphValueType values[kMaxGraphSignatureValues]{};
  u64 value_count = 0u;
  u64 output_count = 0u;
  u64 status_count = 0u;
  bool ok = false;
  const char *reason = "compute_graph_signature_invalid";

  [[nodiscard]] constexpr explicit operator bool() const noexcept { return ok; }
};

struct GraphBufferRef {
  u64 logical_id = 0u;
  BufferRole role = BufferRole::Read;
  BufferInit init = BufferInit::Preserve;
};

struct GraphNode {
  u64 op_hash_hi = 0u;
  u64 op_hash_lo = 0u;
  const GraphBufferRef *buffers = nullptr;
  u64 buffer_count = 0u;
  NodeKind kind = NodeKind::Map;
  u64 primitive_hash_hi = 0u;
  u64 primitive_hash_lo = 0u;
  u64 element_count = 0u;
  GraphControl control{};
};

struct Graph {
  const GraphNode *nodes = nullptr;
  u64 node_count = 0u;
  const u64 *outputs = nullptr;
  u64 output_count = 0u;
  ComputeScalar scalar = ComputeScalar::Lane32;
  ComputeDomain domain = ComputeDomain::I32;
  ComputeFixedFormat fixed_format{};
};

struct GraphCheck {
  u64 graph_id_hi = 0u;
  u64 graph_id_lo = 0u;
  u64 node_count = 0u;
  bool ok = false;
  const char *reason = "compute_graph_invalid";

  [[nodiscard]] constexpr explicit operator bool() const noexcept { return ok; }
};

} // namespace rund::kernel

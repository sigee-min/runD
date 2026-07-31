#pragma once

#include <kernel/program/compute/graph/schema.hpp>

namespace rund::kernel::graph_detail {

struct GraphHash {
  u64 hi = 0u;
  u64 lo = 0u;
};

[[nodiscard]] constexpr u64 Avalanche64(u64 value) noexcept {
  value ^= value >> 30u;
  value *= 0xbf58476d1ce4e5b9ull;
  value ^= value >> 27u;
  value *= 0x94d049bb133111ebull;
  value ^= value >> 31u;
  return value;
}

[[nodiscard]] constexpr GraphHash Mix(const GraphHash hash,
                                      const u64 value) noexcept {
  const u64 mixed = Avalanche64(value + 0x9e3779b97f4a7c15ull + hash.hi +
                                (hash.lo << 6u) + (hash.lo >> 2u));
  return GraphHash{
      .hi = Avalanche64(hash.hi ^ mixed),
      .lo = Avalanche64(hash.lo + mixed + 0x517cc1b727220a95ull),
  };
}

[[nodiscard]] constexpr GraphHash
MixMapNodeIdentity(GraphHash hash, const GraphNode &node) noexcept {
  hash = Mix(hash, node.op_hash_hi);
  hash = Mix(hash, node.op_hash_lo);
  return Mix(hash, node.element_count);
}

[[nodiscard]] constexpr GraphHash
MixCollectiveNodeIdentity(GraphHash hash, const GraphNode &node) noexcept {
  hash = Mix(hash, static_cast<u64>(node.kind));
  hash = Mix(hash, node.primitive_hash_hi);
  hash = Mix(hash, node.primitive_hash_lo);
  return Mix(hash, node.element_count);
}

[[nodiscard]] constexpr GraphHash
MixBufferIdentity(GraphHash hash, const GraphBufferRef &buffer,
                  const u64 buffer_index) noexcept {
  hash = Mix(hash, buffer_index);
  hash = Mix(hash, buffer.logical_id);
  hash = Mix(hash, static_cast<u64>(buffer.role));
  if (buffer.init == BufferInit::Preserve) {
    return hash;
  }
  hash = Mix(hash, 0x627566666572696eull); // "bufferin"
  return Mix(hash, static_cast<u64>(buffer.init));
}

[[nodiscard]] constexpr GraphHash
MixControlIdentity(GraphHash hash, const GraphControl &control) noexcept {
  // Preserve every pre-control graph identity. Default control is absence,
  // not a new sequence of nine zero-valued schema fields. Active control is
  // placed in its own identity domain so node fields cannot alias the
  // extension.
  if (!control.has_count() && !control.has_predicate()) {
    return hash;
  }
  hash = Mix(hash, 0x636f6e74726f6c01ull); // "control" schema v1
  hash = Mix(hash, static_cast<u64>(control.count_source));
  hash = Mix(hash, control.count_binding);
  hash = Mix(hash, control.count_byte_offset);
  hash = Mix(hash, control.capacity);
  hash = Mix(hash, static_cast<u64>(control.predicate_source));
  hash = Mix(hash, control.predicate_binding);
  hash = Mix(hash, control.predicate_byte_offset);
  hash = Mix(hash, control.predicate_expected);
  return Mix(hash, control.iteration);
}

[[nodiscard]] constexpr GraphHash
MixNodeIdentity(GraphHash hash, const GraphNode &node,
                const u64 node_index) noexcept {
  hash = Mix(hash, node_index);
  if (node.kind == NodeKind::Map) {
    hash = MixMapNodeIdentity(hash, node);
  } else {
    hash = MixCollectiveNodeIdentity(hash, node);
  }
  hash = Mix(hash, node.buffer_count);
  for (u64 buffer_index = 0u; buffer_index < node.buffer_count;
       ++buffer_index) {
    hash = MixBufferIdentity(hash, node.buffers[buffer_index], buffer_index);
  }
  return MixControlIdentity(hash, node.control);
}

[[nodiscard]] constexpr GraphHash HashGraph(const Graph &graph) noexcept {
  GraphHash hash{
      .hi = 0x6a09e667f3bcc909ull,
      .lo = 0xbb67ae8584caa73bull,
  };

  hash = Mix(hash, static_cast<u64>(graph.scalar));
  hash = Mix(hash, graph.fixed_format.integer_bits);
  hash = Mix(hash, graph.fixed_format.fraction_bits);
  hash = Mix(hash, static_cast<u64>(graph.fixed_format.rounding));
  hash = Mix(hash, static_cast<u64>(graph.fixed_format.overflow));
  hash = Mix(hash, static_cast<u64>(graph.fixed_format.approximation));
  if (graph.domain != ComputeDomain::Fixed) {
    hash = Mix(hash, 0x646f6d61696e0001ull);
    hash = Mix(hash, static_cast<u64>(graph.domain));
  }
  hash = Mix(hash, graph.node_count);
  for (u64 node_index = 0u; node_index < graph.node_count; ++node_index) {
    hash = MixNodeIdentity(hash, graph.nodes[node_index], node_index);
  }
  if (graph.output_count != 0u) {
    hash = Mix(hash, graph.output_count);
    for (u64 output_index = 0u; output_index < graph.output_count;
         ++output_index) {
      hash = Mix(hash, output_index);
      hash = Mix(hash, graph.outputs[output_index]);
    }
  }

  if (hash.hi == 0u && hash.lo == 0u) {
    hash.lo = 0x9e3779b97f4a7c15ull;
  }
  return hash;
}

} // namespace rund::kernel::graph_detail

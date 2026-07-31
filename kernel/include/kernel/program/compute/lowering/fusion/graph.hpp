#pragma once

#include <kernel/program/compute/fusion.hpp>
#include <kernel/program/compute/lowering/parse.hpp>

namespace rund::kernel {
namespace compute_lowering_detail {

[[nodiscard]] inline u32 CountBindingsOfKind(const ParsedIR &parsed,
                                             const u8 kind) noexcept {
  u32 count = 0u;
  for (const ParsedBinding &binding : parsed.bindings) {
    if (binding.kind == kind) {
      ++count;
    }
  }
  return count;
}

[[nodiscard]] inline u32
CountGraphBuffersOfRole(const GraphNode &node, const BufferRole role) noexcept {
  u32 count = 0u;
  for (u64 index = 0u; index < node.buffer_count; ++index) {
    if (node.buffers[index].role == role) {
      ++count;
    }
  }
  return count;
}

[[nodiscard]] inline u32 BindingIndexForOrdinal(const ParsedIR &parsed,
                                                const u8 kind,
                                                const u32 ordinal) noexcept {
  u32 seen = 0u;
  for (std::size_t index = 0u; index < parsed.bindings.size(); ++index) {
    if (parsed.bindings[index].kind != kind) {
      continue;
    }
    if (seen == ordinal) {
      return static_cast<u32>(index);
    }
    ++seen;
  }
  return static_cast<u32>(parsed.bindings.size());
}

[[nodiscard]] inline u32
GraphReadOrdinalForLogicalId(const GraphNode &node,
                             const u64 logical_id) noexcept {
  u32 ordinal = 0u;
  for (u64 index = 0u; index < node.buffer_count; ++index) {
    const GraphBufferRef &buffer = node.buffers[index];
    if (buffer.role != BufferRole::Read) {
      continue;
    }
    if (buffer.logical_id == logical_id) {
      return ordinal;
    }
    ++ordinal;
  }
  return static_cast<u32>(node.buffer_count);
}

[[nodiscard]] inline u64
IntermediateLogicalId(const Graph &graph, const u64 left_node_index) noexcept {
  if (graph.nodes == nullptr || left_node_index + 1u >= graph.node_count) {
    return 0u;
  }
  const GraphNode &left = graph.nodes[left_node_index];
  const GraphNode &right = graph.nodes[left_node_index + 1u];
  for (u64 left_index = 0u; left_index < left.buffer_count; ++left_index) {
    const GraphBufferRef &left_buffer = left.buffers[left_index];
    if (left_buffer.role != BufferRole::Write) {
      continue;
    }
    for (u64 right_index = 0u; right_index < right.buffer_count;
         ++right_index) {
      const GraphBufferRef &right_buffer = right.buffers[right_index];
      if (right_buffer.role == BufferRole::Read &&
          right_buffer.logical_id == left_buffer.logical_id) {
        return left_buffer.logical_id;
      }
    }
  }
  return 0u;
}

[[nodiscard]] inline u32 WriteNodeCount(const ParsedIR &parsed) noexcept {
  u32 write_count = 0u;
  for (const ParsedNode &node : parsed.nodes) {
    if (node.op == static_cast<u8>(IrOp::Write)) {
      ++write_count;
    }
  }
  return write_count;
}

} // namespace compute_lowering_detail

} // namespace rund::kernel

#pragma once

#include <kernel/program/compute/graph/identity.hpp>
#include <kernel/program/compute/limit.hpp>

namespace rund::kernel {

namespace graph_detail {

[[nodiscard]] constexpr bool
DomainMatchesScalar(const ComputeDomain domain,
                    const ComputeScalar scalar) noexcept {
  if (scalar == ComputeScalar::Lane32) {
    return domain == ComputeDomain::I32 || domain == ComputeDomain::U32 ||
           domain == ComputeDomain::Fixed;
  }
  return scalar == ComputeScalar::Lane64 &&
         (domain == ComputeDomain::I64 || domain == ComputeDomain::U64 ||
          domain == ComputeDomain::Fixed);
}

[[nodiscard]] constexpr bool KnownRole(const BufferRole role) noexcept {
  return role == BufferRole::Read || role == BufferRole::Write;
}

[[nodiscard]] constexpr bool KnownInit(const BufferInit init) noexcept {
  return init == BufferInit::Preserve || init == BufferInit::Zero;
}

[[nodiscard]] constexpr GraphCheck
RejectGraph(const Graph &graph, const char *const reason) noexcept {
  return GraphCheck{
      .node_count = graph.node_count,
      .reason = reason,
  };
}

[[nodiscard]] constexpr const char *
ValidateGraphIdentityHeader(const Graph &graph) noexcept {
  if (!ComputeScalarValid(graph.scalar) ||
      !DomainMatchesScalar(graph.domain, graph.scalar) ||
      (graph.domain == ComputeDomain::Fixed
           ? !ComputeFixedFormatValid(graph.scalar, graph.fixed_format)
           : !ComputeFixedFormatAbsent(graph.fixed_format))) {
    return "compute_graph_numeric_invalid";
  }
  if (graph.node_count > kMaxGraphNodeCount) {
    return "compute_graph_node_count_invalid";
  }
  if (graph.output_count > kMaxGraphOutputCount ||
      (graph.output_count != 0u && graph.outputs == nullptr)) {
    return "compute_graph_output_invalid";
  }
  for (u64 output = 0u; output < graph.output_count; ++output) {
    if (graph.outputs[output] == 0u) {
      return "compute_graph_output_invalid";
    }
  }
  return graph.node_count != 0u && graph.nodes == nullptr
             ? "compute_graph_node_invalid"
             : nullptr;
}

[[nodiscard]] constexpr const char *
ValidateGraphHeader(const Graph &graph) noexcept {
  if (graph.node_count == 0u) {
    return "compute_graph_empty";
  }
  return ValidateGraphIdentityHeader(graph);
}

[[nodiscard]] constexpr const char *
ValidateMapNodeIdentity(const GraphNode &node) noexcept {
  if (node.op_hash_hi == 0u && node.op_hash_lo == 0u) {
    return "compute_graph_node_invalid";
  }
  if (node.primitive_hash_hi != 0u || node.primitive_hash_lo != 0u ||
      node.element_count == 0u) {
    return "compute_graph_primitive_invalid";
  }
  return nullptr;
}

[[nodiscard]] constexpr const char *
ValidateMapNodeRecipeIdentity(const GraphNode &node) noexcept {
  if (node.op_hash_hi == 0u && node.op_hash_lo == 0u) {
    return "compute_graph_node_invalid";
  }
  return node.primitive_hash_hi != 0u || node.primitive_hash_lo != 0u
             ? "compute_graph_primitive_invalid"
             : nullptr;
}

[[nodiscard]] constexpr const char *
ValidateCollectiveNodeIdentity(const GraphNode &node) noexcept {
  if ((node.primitive_hash_hi == 0u && node.primitive_hash_lo == 0u) ||
      node.element_count == 0u || node.op_hash_hi != 0u ||
      node.op_hash_lo != 0u) {
    return "compute_graph_primitive_invalid";
  }
  return nullptr;
}

[[nodiscard]] constexpr const char *
ValidateCollectiveNodeRecipeIdentity(const GraphNode &node) noexcept {
  return (node.primitive_hash_hi == 0u && node.primitive_hash_lo == 0u) ||
                 node.op_hash_hi != 0u || node.op_hash_lo != 0u
             ? "compute_graph_primitive_invalid"
             : nullptr;
}

[[nodiscard]] constexpr const char *
ValidateNodeIdentity(const GraphNode &node) noexcept {
  if (!NodeKindValid(node.kind)) {
    return "compute_graph_node_invalid";
  }
  return node.kind == NodeKind::Map ? ValidateMapNodeIdentity(node)
                                    : ValidateCollectiveNodeIdentity(node);
}

[[nodiscard]] constexpr const char *
ValidateNodeRecipeIdentity(const GraphNode &node) noexcept {
  if (!NodeKindValid(node.kind)) {
    return "compute_graph_node_invalid";
  }
  return node.kind == NodeKind::Map
             ? ValidateMapNodeRecipeIdentity(node)
             : ValidateCollectiveNodeRecipeIdentity(node);
}

[[nodiscard]] constexpr const char *
ValidateNodeBuffers(const GraphNode &node) noexcept {
  if (node.buffer_count > kMaxGraphBuffersPerNode) {
    return "compute_graph_buffer_count_invalid";
  }
  if (node.buffer_count == 0u || node.buffers == nullptr) {
    return "compute_graph_buffer_invalid";
  }
  for (u64 buffer_index = 0u; buffer_index < node.buffer_count;
       ++buffer_index) {
    const GraphBufferRef &buffer = node.buffers[buffer_index];
    if (buffer.logical_id == 0u || !KnownRole(buffer.role) ||
        !KnownInit(buffer.init) ||
        (buffer.role == BufferRole::Read &&
         buffer.init != BufferInit::Preserve)) {
      return "compute_graph_buffer_invalid";
    }
  }
  if (!node.control.valid(node.buffer_count) ||
      (node.control.has_count() &&
       (node.control.capacity != node.element_count ||
        node.buffers[node.control.count_binding].role != BufferRole::Read)) ||
      (node.control.has_predicate() &&
       node.buffers[node.control.predicate_binding].role != BufferRole::Read)) {
    return "compute_graph_control_invalid";
  }
  return nullptr;
}

[[nodiscard]] constexpr const char *
ValidateNode(const GraphNode &node) noexcept {
  if (const char *const reason = ValidateNodeIdentity(node);
      reason != nullptr) {
    return reason;
  }
  return ValidateNodeBuffers(node);
}

} // namespace graph_detail

[[nodiscard]] constexpr GraphCheck ValidateGraph(const Graph &graph) noexcept {
  if (const char *const reason = graph_detail::ValidateGraphHeader(graph);
      reason != nullptr) {
    return graph_detail::RejectGraph(graph, reason);
  }
  for (u64 node_index = 0u; node_index < graph.node_count; ++node_index) {
    if (const char *const reason =
            graph_detail::ValidateNode(graph.nodes[node_index]);
        reason != nullptr) {
      return graph_detail::RejectGraph(graph, reason);
    }
  }
  for (u64 output = 0u; output < graph.output_count; ++output) {
    bool written = false;
    for (u64 node = 0u; node < graph.node_count && !written; ++node) {
      for (u64 buffer = 0u; buffer < graph.nodes[node].buffer_count; ++buffer) {
        const GraphBufferRef &ref = graph.nodes[node].buffers[buffer];
        written = ref.logical_id == graph.outputs[output] &&
                  ref.role == BufferRole::Write;
        if (written) {
          break;
        }
      }
    }
    if (!written) {
      return graph_detail::RejectGraph(graph, "compute_graph_output_invalid");
    }
  }

  const graph_detail::GraphHash hash = graph_detail::HashGraph(graph);
  return GraphCheck{
      .graph_id_hi = hash.hi,
      .graph_id_lo = hash.lo,
      .node_count = graph.node_count,
      .ok = true,
      .reason = "ok",
  };
}

// Zero-work product programs have no executable nodes, but still need the
// exact same policy/output identity law as executable graphs. This identity
// validator is deliberately separate from ValidateGraph: execution admission
// continues to reject an empty graph descriptor.
[[nodiscard]] constexpr GraphCheck
ValidateGraphIdentity(const Graph &graph) noexcept {
  if (graph.node_count == 0u) {
    return graph_detail::RejectGraph(graph, "compute_graph_empty");
  }
  if (const char *const reason =
          graph_detail::ValidateGraphIdentityHeader(graph);
      reason != nullptr) {
    return graph_detail::RejectGraph(graph, reason);
  }
  for (u64 node_index = 0u; node_index < graph.node_count; ++node_index) {
    const GraphNode &node = graph.nodes[node_index];
    if (const char *const reason =
            graph_detail::ValidateNodeRecipeIdentity(node);
        reason != nullptr) {
      return graph_detail::RejectGraph(graph, reason);
    }
    if (const char *const reason = graph_detail::ValidateNodeBuffers(node);
        reason != nullptr) {
      return graph_detail::RejectGraph(graph, reason);
    }
  }
  for (u64 output = 0u; output < graph.output_count; ++output) {
    bool written = false;
    for (u64 node = 0u; node < graph.node_count && !written; ++node) {
      for (u64 buffer = 0u; buffer < graph.nodes[node].buffer_count; ++buffer) {
        const GraphBufferRef &ref = graph.nodes[node].buffers[buffer];
        written = ref.logical_id == graph.outputs[output] &&
                  ref.role == BufferRole::Write;
        if (written) {
          break;
        }
      }
    }
    if (!written) {
      return graph_detail::RejectGraph(graph, "compute_graph_output_invalid");
    }
  }
  const graph_detail::GraphHash hash = graph_detail::HashGraph(graph);
  return GraphCheck{
      .graph_id_hi = hash.hi,
      .graph_id_lo = hash.lo,
      .node_count = 0u,
      .ok = true,
      .reason = "ok",
  };
}
} // namespace rund::kernel

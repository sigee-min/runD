#pragma once

#include "test/compute/fixed.hpp"

#include "test/assert.hpp"

#include <kernel/program/compute/graph/schema.hpp>
#include <kernel/program/compute/graph/identity.hpp>
#include <kernel/program/compute/graph/validation.hpp>

namespace program_compute_contract {

struct TwoMapGraphOptions {
  rund::kernel::u64 first_op_hash_hi = 0x1020304050607080u;
  rund::kernel::u64 first_op_hash_lo = 0x8877665544332211u;
  rund::kernel::u64 first_read_id = 11u;
  rund::kernel::u64 first_write_id = 21u;
  rund::kernel::BufferRole first_read_role = rund::kernel::BufferRole::Read;
  rund::kernel::BufferRole first_write_role = rund::kernel::BufferRole::Write;
  rund::kernel::ComputeScalar scalar = rund::kernel::ComputeScalar::Lane32;
  rund::kernel::ComputeDomain domain = rund::kernel::ComputeDomain::Fixed;
  rund::kernel::ComputeFixedFormat fixed_format{};
  rund::kernel::u64 element_count = 1024u;
  bool reorder_first_buffers = false;
  bool reorder_nodes = false;
};

[[nodiscard]] inline bool GraphIdsDiffer(const rund::kernel::GraphCheck &lhs,
                                         const rund::kernel::GraphCheck &rhs) {
  return lhs.graph_id_hi != rhs.graph_id_hi ||
         lhs.graph_id_lo != rhs.graph_id_lo;
}

[[nodiscard]] inline rund::kernel::GraphCheck
MakeTwoMapGraph(const TwoMapGraphOptions options = {}) noexcept {
  const rund::kernel::GraphBufferRef first_read{
      .logical_id = options.first_read_id,
      .role = options.first_read_role,
  };
  const rund::kernel::GraphBufferRef first_write{
      .logical_id = options.first_write_id,
      .role = options.first_write_role,
  };
  const rund::kernel::GraphBufferRef first_buffers_ordered[] = {
      first_read,
      first_write,
  };
  const rund::kernel::GraphBufferRef first_buffers_reordered[] = {
      first_write,
      first_read,
  };
  const rund::kernel::GraphBufferRef second_buffers[] = {
      {.logical_id = 21u, .role = rund::kernel::BufferRole::Read},
      {.logical_id = 31u, .role = rund::kernel::BufferRole::Write},
  };
  const rund::kernel::GraphBufferRef *const first_buffers =
      options.reorder_first_buffers ? first_buffers_reordered
                                    : first_buffers_ordered;
  const rund::kernel::GraphNode first_node{
      .op_hash_hi = options.first_op_hash_hi,
      .op_hash_lo = options.first_op_hash_lo,
      .buffers = first_buffers,
      .buffer_count = 2u,
      .element_count = options.element_count,
  };
  const rund::kernel::GraphNode second_node{
      .op_hash_hi = 0x2122232425262728u,
      .op_hash_lo = 0x9988776655443322u,
      .buffers = second_buffers,
      .buffer_count = 2u,
      .element_count = options.element_count,
  };
  const rund::kernel::GraphNode nodes_ordered[] = {
      first_node,
      second_node,
  };
  const rund::kernel::GraphNode nodes_reordered[] = {
      second_node,
      first_node,
  };
  const rund::kernel::GraphNode *const nodes =
      options.reorder_nodes ? nodes_reordered : nodes_ordered;
  const rund::kernel::Graph graph{
      .nodes = nodes,
      .node_count = 2u,
      .scalar = options.scalar,
      .domain = options.domain,
      .fixed_format =
          options.domain == rund::kernel::ComputeDomain::Fixed
              ? (rund::kernel::ComputeFixedFormatAbsent(options.fixed_format)
                     ? test::FixedFormatForLane(options.scalar)
                     : options.fixed_format)
              : rund::kernel::ComputeFixedFormat{},
  };

  return rund::kernel::ValidateGraph(graph);
}

int RunGraphIdentityContract();
int RunGraphAdmissionContract();
int RunGraphSignatureContract();
int RunBoundedGraphSignatureContract();
int RunGraphRejectionContract();

} // namespace program_compute_contract

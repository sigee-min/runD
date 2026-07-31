#pragma once

#include "policy.hpp"

namespace program_compute_contract::fusion_support {

struct LinearGraphFixture {
  rund::kernel::GraphBufferRef first_buffers[2] = {
      {.logical_id = 11u, .role = rund::kernel::BufferRole::Read},
      {.logical_id = 21u, .role = rund::kernel::BufferRole::Write},
  };
  rund::kernel::GraphBufferRef second_buffers[2] = {
      {.logical_id = 21u, .role = rund::kernel::BufferRole::Read},
      {.logical_id = 31u, .role = rund::kernel::BufferRole::Write},
  };
  rund::kernel::GraphNode nodes[2]{};
  rund::kernel::Graph graph{};

  explicit LinearGraphFixture(
      const TestOpHash second_op = kSecondOp) noexcept {
    nodes[0] = rund::kernel::GraphNode{
        .op_hash_hi = kFirstOp.op_hash_hi,
        .op_hash_lo = kFirstOp.op_hash_lo,
        .buffers = first_buffers,
        .buffer_count = 2u,
        .element_count = 1024u,
    };
    nodes[1] = rund::kernel::GraphNode{
        .op_hash_hi = second_op.op_hash_hi,
        .op_hash_lo = second_op.op_hash_lo,
        .buffers = second_buffers,
        .buffer_count = 2u,
        .element_count = 1024u,
    };
    graph = rund::kernel::Graph{
        .nodes = nodes,
        .node_count = 2u,
        .scalar = rund::kernel::ComputeScalar::Lane32,
    };
  }
};

struct MultiWriteProducerGraphFixture {
  rund::kernel::GraphBufferRef first_buffers[3] = {
      {.logical_id = 11u, .role = rund::kernel::BufferRole::Read},
      {.logical_id = 21u, .role = rund::kernel::BufferRole::Write},
      {.logical_id = 22u, .role = rund::kernel::BufferRole::Write},
  };
  rund::kernel::GraphBufferRef second_buffers[2] = {
      {.logical_id = 21u, .role = rund::kernel::BufferRole::Read},
      {.logical_id = 31u, .role = rund::kernel::BufferRole::Write},
  };
  rund::kernel::GraphNode nodes[2]{};
  rund::kernel::Graph graph{};

  MultiWriteProducerGraphFixture() noexcept {
    nodes[0] = rund::kernel::GraphNode{
        .op_hash_hi = kFirstOp.op_hash_hi,
        .op_hash_lo = kFirstOp.op_hash_lo,
        .buffers = first_buffers,
        .buffer_count = 3u,
        .element_count = 1024u,
    };
    nodes[1] = rund::kernel::GraphNode{
        .op_hash_hi = kSecondOp.op_hash_hi,
        .op_hash_lo = kSecondOp.op_hash_lo,
        .buffers = second_buffers,
        .buffer_count = 2u,
        .element_count = 1024u,
    };
    graph = rund::kernel::Graph{
        .nodes = nodes,
        .node_count = 2u,
        .scalar = rund::kernel::ComputeScalar::Lane32,
    };
  }
};

struct ThreeNodeChainGraphFixture {
  rund::kernel::GraphBufferRef first_buffers[2] = {
      {.logical_id = 11u, .role = rund::kernel::BufferRole::Read},
      {.logical_id = 21u, .role = rund::kernel::BufferRole::Write},
  };
  rund::kernel::GraphBufferRef second_buffers[2] = {
      {.logical_id = 21u, .role = rund::kernel::BufferRole::Read},
      {.logical_id = 31u, .role = rund::kernel::BufferRole::Write},
  };
  rund::kernel::GraphBufferRef third_buffers[2] = {
      {.logical_id = 31u, .role = rund::kernel::BufferRole::Read},
      {.logical_id = 41u, .role = rund::kernel::BufferRole::Write},
  };
  rund::kernel::GraphNode nodes[3]{};
  rund::kernel::Graph graph{};

  ThreeNodeChainGraphFixture() noexcept {
    nodes[0] = rund::kernel::GraphNode{
        .op_hash_hi = kFirstOp.op_hash_hi,
        .op_hash_lo = kFirstOp.op_hash_lo,
        .buffers = first_buffers,
        .buffer_count = 2u,
        .element_count = 1024u,
    };
    nodes[1] = rund::kernel::GraphNode{
        .op_hash_hi = kSecondOp.op_hash_hi,
        .op_hash_lo = kSecondOp.op_hash_lo,
        .buffers = second_buffers,
        .buffer_count = 2u,
        .element_count = 1024u,
    };
    nodes[2] = rund::kernel::GraphNode{
        .op_hash_hi = kThirdOp.op_hash_hi,
        .op_hash_lo = kThirdOp.op_hash_lo,
        .buffers = third_buffers,
        .buffer_count = 2u,
        .element_count = 1024u,
    };
    graph = rund::kernel::Graph{
        .nodes = nodes,
        .node_count = 3u,
        .scalar = rund::kernel::ComputeScalar::Lane32,
    };
  }
};

struct BranchedConsumerGraphFixture {
  rund::kernel::GraphBufferRef first_buffers[2] = {
      {.logical_id = 11u, .role = rund::kernel::BufferRole::Read},
      {.logical_id = 21u, .role = rund::kernel::BufferRole::Write},
  };
  rund::kernel::GraphBufferRef second_buffers[2] = {
      {.logical_id = 21u, .role = rund::kernel::BufferRole::Read},
      {.logical_id = 31u, .role = rund::kernel::BufferRole::Write},
  };
  rund::kernel::GraphBufferRef third_buffers[2] = {
      {.logical_id = 21u, .role = rund::kernel::BufferRole::Read},
      {.logical_id = 41u, .role = rund::kernel::BufferRole::Write},
  };
  rund::kernel::GraphNode nodes[3]{};
  rund::kernel::Graph graph{};

  BranchedConsumerGraphFixture() noexcept {
    nodes[0] = rund::kernel::GraphNode{
        .op_hash_hi = kFirstOp.op_hash_hi,
        .op_hash_lo = kFirstOp.op_hash_lo,
        .buffers = first_buffers,
        .buffer_count = 2u,
        .element_count = 1024u,
    };
    nodes[1] = rund::kernel::GraphNode{
        .op_hash_hi = kSecondOp.op_hash_hi,
        .op_hash_lo = kSecondOp.op_hash_lo,
        .buffers = second_buffers,
        .buffer_count = 2u,
        .element_count = 1024u,
    };
    nodes[2] = rund::kernel::GraphNode{
        .op_hash_hi = kSecondOp.op_hash_hi,
        .op_hash_lo = kSecondOp.op_hash_lo,
        .buffers = third_buffers,
        .buffer_count = 2u,
        .element_count = 1024u,
    };
    graph = rund::kernel::Graph{
        .nodes = nodes,
        .node_count = 3u,
        .scalar = rund::kernel::ComputeScalar::Lane32,
    };
  }
};

[[nodiscard]] inline bool SameId(const rund::kernel::u64 lhs_hi,
                                 const rund::kernel::u64 lhs_lo,
                                 const rund::kernel::u64 rhs_hi,
                                 const rund::kernel::u64 rhs_lo) noexcept {
  return lhs_hi == rhs_hi && lhs_lo == rhs_lo;
}

} // namespace program_compute_contract::fusion_support

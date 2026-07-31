#pragma once

#include "policy.hpp"

namespace program_compute_contract::fusion_support {

[[nodiscard]] inline rund::kernel::ComputeIR
BuildManyReadIr(const std::string_view name,
                const rund::kernel::u32 read_count) {
  std::vector<rund::kernel::u8> bytes{};
  AppendBytes(bytes, "rund.compute.ir");
  AppendBytes(bytes, name);
  AppendU8(bytes, kI32NumericMode);
  AppendIntegerNumericPolicy(bytes);
  AppendU32(bytes, read_count + 1u);
  for (rund::kernel::u32 read = 0u; read < read_count; ++read) {
    AppendBinding(bytes, 2u, "read" + std::to_string(read));
  }
  AppendBinding(bytes, 3u, "out");

  const rund::kernel::u32 add_count = read_count - 1u;
  AppendU32(bytes, read_count + add_count + 1u);
  for (rund::kernel::u32 read = 0u; read < read_count; ++read) {
    AppendNode(bytes, rund::kernel::IrOp::Read, 0u, 0u, read);
  }
  rund::kernel::u32 value_node = read_count == 0u ? 0u : 1u;
  for (rund::kernel::u32 read = 1u; read < read_count; ++read) {
    AppendNode(bytes, rund::kernel::IrOp::Add, value_node, read + 1u, 0u);
    value_node = read_count + read;
  }
  AppendNode(bytes, rund::kernel::IrOp::Write, value_node, 0u, read_count);
  return IrFromBytes(std::move(bytes));
}

struct OversizedFusedBindingFixture {
  static constexpr rund::kernel::u32 kFirstReadCount = 33u;
  static constexpr rund::kernel::u32 kSecondReadCount = 33u;

  rund::kernel::ComputeIR first =
      BuildManyReadIr("oversized-first", kFirstReadCount);
  rund::kernel::ComputeIR second =
      BuildManyReadIr("oversized-second", kSecondReadCount);
  std::array<rund::kernel::GraphBufferRef, kFirstReadCount + 1u>
      first_buffers{};
  std::array<rund::kernel::GraphBufferRef, kSecondReadCount + 1u>
      second_buffers{};
  rund::kernel::GraphNode nodes[2]{};
  rund::kernel::Graph graph{};
  rund::kernel::FusionNodePolicy fusion_nodes[2]{};
  rund::kernel::FusionPolicy policy{};

  OversizedFusedBindingFixture() {
    for (rund::kernel::u32 read = 0u; read < kFirstReadCount; ++read) {
      first_buffers[read] = rund::kernel::GraphBufferRef{
          .logical_id = 1000u + read,
          .role = rund::kernel::BufferRole::Read,
      };
    }
    first_buffers[kFirstReadCount] = rund::kernel::GraphBufferRef{
        .logical_id = 21u,
        .role = rund::kernel::BufferRole::Write,
    };
    second_buffers[0] = rund::kernel::GraphBufferRef{
        .logical_id = 21u,
        .role = rund::kernel::BufferRole::Read,
    };
    for (rund::kernel::u32 read = 1u; read < kSecondReadCount; ++read) {
      second_buffers[read] = rund::kernel::GraphBufferRef{
          .logical_id = 2000u + read,
          .role = rund::kernel::BufferRole::Read,
      };
    }
    second_buffers[kSecondReadCount] = rund::kernel::GraphBufferRef{
        .logical_id = 31u,
        .role = rund::kernel::BufferRole::Write,
    };
    nodes[0] = rund::kernel::GraphNode{
        .op_hash_hi = first.op_hash_hi,
        .op_hash_lo = first.op_hash_lo,
        .buffers = first_buffers.data(),
        .buffer_count = first_buffers.size(),
        .element_count = 1024u,
    };
    nodes[1] = rund::kernel::GraphNode{
        .op_hash_hi = second.op_hash_hi,
        .op_hash_lo = second.op_hash_lo,
        .buffers = second_buffers.data(),
        .buffer_count = second_buffers.size(),
        .element_count = 1024u,
    };
    graph = rund::kernel::Graph{
        .nodes = nodes,
        .node_count = 2u,
        .scalar = rund::kernel::ComputeScalar::Lane32,
        .domain = rund::kernel::ComputeDomain::I32,
    };
    fusion_nodes[0] = PolicyNode(first);
    fusion_nodes[1] = PolicyNode(second);
    policy = SupportedPolicy(fusion_nodes, 2u);
  }
};

} // namespace program_compute_contract::fusion_support

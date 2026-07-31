#include "local.hpp"

namespace program_compute_contract {
namespace {

int InvalidBuffer() {
  const rund::kernel::GraphBufferRef buffers[] = {
      {.logical_id = 0u, .role = rund::kernel::BufferRole::Read},
      {.logical_id = 21u, .role = rund::kernel::BufferRole::Write},
  };
  const rund::kernel::GraphNode nodes[] = {
      {.op_hash_hi = 0x1020304050607080u,
       .op_hash_lo = 0x8877665544332211u,
       .buffers = buffers,
       .buffer_count = 2u,
       .element_count = 1u},
  };
  const rund::kernel::Graph graph{
      .nodes = nodes,
      .node_count = 1u,
      .scalar = rund::kernel::ComputeScalar::Lane32,
  };
  const auto check = rund::kernel::ValidateGraph(graph);

  TEST_ASSERT(!check.ok);
  TEST_ASSERT(std::string_view{check.reason} == "compute_graph_buffer_invalid");
  return 0;
}

int MissingBuffers() {
  const rund::kernel::GraphNode nodes[] = {
      {.op_hash_hi = 0x1020304050607080u,
       .op_hash_lo = 0x8877665544332211u,
       .buffers = nullptr,
       .buffer_count = 1u,
       .element_count = 1u},
  };
  const rund::kernel::Graph graph{
      .nodes = nodes,
      .node_count = 1u,
      .scalar = rund::kernel::ComputeScalar::Lane32,
  };
  const auto check = rund::kernel::ValidateGraph(graph);

  TEST_ASSERT(!check.ok);
  TEST_ASSERT(std::string_view{check.reason} == "compute_graph_buffer_invalid");
  return 0;
}

int HugeBufferCount() {
  const rund::kernel::GraphBufferRef buffers[] = {
      {.logical_id = 11u, .role = rund::kernel::BufferRole::Read},
  };
  const rund::kernel::GraphNode nodes[] = {
      {.op_hash_hi = 0x1020304050607080u,
       .op_hash_lo = 0x8877665544332211u,
       .buffers = buffers,
       .buffer_count = rund::kernel::kMaxGraphBuffersPerNode + 1u,
       .element_count = 1u},
  };
  const rund::kernel::Graph graph{
      .nodes = nodes,
      .node_count = 1u,
      .scalar = rund::kernel::ComputeScalar::Lane32,
  };
  const auto check = rund::kernel::ValidateGraph(graph);

  TEST_ASSERT(!check.ok);
  TEST_ASSERT(std::string_view{check.reason} ==
              "compute_graph_buffer_count_invalid");
  return 0;
}

int ForgedBufferRole() {
  const rund::kernel::GraphBufferRef buffers[] = {
      {.logical_id = 11u, .role = static_cast<rund::kernel::BufferRole>(0xffu)},
  };
  const rund::kernel::GraphNode nodes[] = {
      {.op_hash_hi = 0x1020304050607080u,
       .op_hash_lo = 0x8877665544332211u,
       .buffers = buffers,
       .buffer_count = 1u,
       .element_count = 1u},
  };
  const rund::kernel::Graph graph{
      .nodes = nodes,
      .node_count = 1u,
      .scalar = rund::kernel::ComputeScalar::Lane32,
  };
  const auto check = rund::kernel::ValidateGraph(graph);

  TEST_ASSERT(!check.ok);
  TEST_ASSERT(std::string_view{check.reason} == "compute_graph_buffer_invalid");
  return 0;
}

int InvalidBufferInit() {
  const rund::kernel::GraphBufferRef read_zero[] = {
      {.logical_id = 11u,
       .role = rund::kernel::BufferRole::Read,
       .init = rund::kernel::BufferInit::Zero},
  };
  const rund::kernel::GraphNode node{
      .op_hash_hi = 0x1020304050607080u,
      .op_hash_lo = 0x8877665544332211u,
      .buffers = read_zero,
      .buffer_count = 1u,
      .element_count = 1u,
  };
  const rund::kernel::Graph graph{
      .nodes = &node,
      .node_count = 1u,
      .scalar = rund::kernel::ComputeScalar::Lane32,
  };
  const auto read_check = rund::kernel::ValidateGraph(graph);
  rund::kernel::GraphBufferRef forged = read_zero[0u];
  forged.role = rund::kernel::BufferRole::Write;
  forged.init = static_cast<rund::kernel::BufferInit>(0xffu);
  rund::kernel::GraphNode forged_node = node;
  forged_node.buffers = &forged;
  const rund::kernel::Graph forged_graph{
      .nodes = &forged_node,
      .node_count = 1u,
      .scalar = rund::kernel::ComputeScalar::Lane32,
  };
  const auto forged_check = rund::kernel::ValidateGraph(forged_graph);
  TEST_ASSERT(!read_check.ok);
  TEST_ASSERT(std::string_view{read_check.reason} ==
              "compute_graph_buffer_invalid");
  TEST_ASSERT(!forged_check.ok);
  TEST_ASSERT(std::string_view{forged_check.reason} ==
              "compute_graph_buffer_invalid");
  return 0;
}

} // namespace

int GraphRejectBuffer() {
  if (InvalidBuffer() != 0) {
    return 1;
  }
  if (MissingBuffers() != 0) {
    return 1;
  }
  if (HugeBufferCount() != 0) {
    return 1;
  }
  if (ForgedBufferRole() != 0) {
    return 1;
  }
  return InvalidBufferInit();
}

} // namespace program_compute_contract

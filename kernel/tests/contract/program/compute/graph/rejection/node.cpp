#include "local.hpp"

namespace program_compute_contract {
namespace {

int EmptyProgram() {
  const rund::kernel::Graph graph{};
  const auto check = rund::kernel::ValidateGraph(graph);
  const auto identity_check = rund::kernel::ValidateGraphIdentity(graph);

  TEST_ASSERT(!check.ok);
  TEST_ASSERT(std::string_view{check.reason} == "compute_graph_empty");
  TEST_ASSERT(!identity_check.ok);
  TEST_ASSERT(std::string_view{identity_check.reason} ==
              "compute_graph_empty");
  return 0;
}

int InvalidNode() {
  const rund::kernel::GraphBufferRef buffers[] = {
      {.logical_id = 11u, .role = rund::kernel::BufferRole::Read},
      {.logical_id = 21u, .role = rund::kernel::BufferRole::Write},
  };
  const rund::kernel::GraphNode nodes[] = {
      {.op_hash_hi = 0u,
       .op_hash_lo = 0u,
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
  const auto identity_check = rund::kernel::ValidateGraphIdentity(graph);

  TEST_ASSERT(!check.ok);
  TEST_ASSERT(std::string_view{check.reason} == "compute_graph_node_invalid");
  TEST_ASSERT(!identity_check.ok);
  TEST_ASSERT(std::string_view{identity_check.reason} ==
              "compute_graph_node_invalid");
  return 0;
}

int MissingNodes() {
  const rund::kernel::Graph graph{
      .nodes = nullptr,
      .node_count = 1u,
      .scalar = rund::kernel::ComputeScalar::Lane32,
  };
  const auto check = rund::kernel::ValidateGraph(graph);

  TEST_ASSERT(!check.ok);
  TEST_ASSERT(std::string_view{check.reason} == "compute_graph_node_invalid");
  return 0;
}

int HugeNodeCount() {
  const rund::kernel::GraphBufferRef buffers[] = {
      {.logical_id = 11u, .role = rund::kernel::BufferRole::Read},
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
      .node_count = rund::kernel::kMaxGraphNodeCount + 1u,
      .scalar = rund::kernel::ComputeScalar::Lane32,
  };
  const auto check = rund::kernel::ValidateGraph(graph);

  TEST_ASSERT(!check.ok);
  TEST_ASSERT(std::string_view{check.reason} ==
              "compute_graph_node_count_invalid");
  return 0;
}

} // namespace

int GraphRejectNode() {
  if (EmptyProgram() != 0) {
    return 1;
  }
  if (InvalidNode() != 0) {
    return 1;
  }
  if (MissingNodes() != 0) {
    return 1;
  }
  return HugeNodeCount();
}

} // namespace program_compute_contract

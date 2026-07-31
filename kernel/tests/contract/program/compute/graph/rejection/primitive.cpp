#include "local.hpp"

namespace program_compute_contract {

int GraphRejectPrimitive() {
  const rund::kernel::GraphBufferRef buffers[] = {
      {.logical_id = 11u, .role = rund::kernel::BufferRole::Read},
      {.logical_id = 21u, .role = rund::kernel::BufferRole::Write},
  };
  const rund::kernel::GraphNode missing_descriptor{
      .buffers = buffers,
      .buffer_count = 2u,
      .kind = rund::kernel::NodeKind::Sort,
      .primitive_hash_hi = 0u,
      .primitive_hash_lo = 0u,
      .element_count = 1024u,
  };
  const rund::kernel::Graph graph{
      .nodes = &missing_descriptor,
      .node_count = 1u,
      .scalar = rund::kernel::ComputeScalar::Lane32,
  };
  const auto descriptor_check = rund::kernel::ValidateGraph(graph);
  const auto descriptor_identity_check =
      rund::kernel::ValidateGraphIdentity(graph);
  const rund::kernel::GraphNode zero_count{
      .buffers = buffers,
      .buffer_count = 2u,
      .kind = rund::kernel::NodeKind::Scan,
      .primitive_hash_hi = 0x1111222233334444u,
      .primitive_hash_lo = 0x5555666677778888u,
      .element_count = 0u,
  };
  const rund::kernel::Graph zero_count_graph{
      .nodes = &zero_count,
      .node_count = 1u,
      .scalar = rund::kernel::ComputeScalar::Lane32,
  };
  const auto count_check = rund::kernel::ValidateGraph(zero_count_graph);
  const auto count_identity_check =
      rund::kernel::ValidateGraphIdentity(zero_count_graph);
  const rund::kernel::GraphNode mixed_map{
      .op_hash_hi = 0x1111222233334444u,
      .op_hash_lo = 0x5555666677778888u,
      .buffers = buffers,
      .buffer_count = 2u,
      .kind = rund::kernel::NodeKind::Map,
      .primitive_hash_hi = 0x9999aaaabbbbccccu,
      .element_count = 1024u,
  };
  const rund::kernel::Graph mixed_map_graph{
      .nodes = &mixed_map,
      .node_count = 1u,
      .scalar = rund::kernel::ComputeScalar::Lane32,
  };
  const auto mixed_map_check =
      rund::kernel::ValidateGraph(mixed_map_graph);
  const rund::kernel::GraphNode reserved_kind{
      .buffers = buffers,
      .buffer_count = 2u,
      .kind = static_cast<rund::kernel::NodeKind>(5u),
      .primitive_hash_hi = 0x1111222233334444u,
      .primitive_hash_lo = 0x5555666677778888u,
      .element_count = 1024u,
  };
  const rund::kernel::Graph reserved_kind_graph{
      .nodes = &reserved_kind,
      .node_count = 1u,
      .scalar = rund::kernel::ComputeScalar::Lane32,
  };
  const auto reserved_kind_check =
      rund::kernel::ValidateGraph(reserved_kind_graph);

  TEST_ASSERT(!descriptor_check.ok);
  TEST_ASSERT(std::string_view{descriptor_check.reason} ==
              "compute_graph_primitive_invalid");
  TEST_ASSERT(!descriptor_identity_check.ok);
  TEST_ASSERT(std::string_view{descriptor_identity_check.reason} ==
              "compute_graph_primitive_invalid");
  TEST_ASSERT(!count_check.ok);
  TEST_ASSERT(std::string_view{count_check.reason} ==
              "compute_graph_primitive_invalid");
  TEST_ASSERT(count_identity_check.ok);
  TEST_ASSERT(!mixed_map_check.ok);
  TEST_ASSERT(std::string_view{mixed_map_check.reason} ==
              "compute_graph_primitive_invalid");
  TEST_ASSERT(!reserved_kind_check.ok);
  TEST_ASSERT(std::string_view{reserved_kind_check.reason} ==
              "compute_graph_node_invalid");
  return 0;
}

} // namespace program_compute_contract

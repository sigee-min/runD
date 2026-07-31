#include "local.hpp"

namespace program_compute_contract {
namespace {

int UnsupportedScalar() {
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
      .node_count = 1u,
      .scalar = static_cast<rund::kernel::ComputeScalar>(0xffu),
  };
  const auto check = rund::kernel::ValidateGraph(graph);

  TEST_ASSERT(!check.ok);
  TEST_ASSERT(std::string_view{check.reason} ==
              "compute_graph_numeric_invalid");
  return 0;
}

int NonFixedFormatMustBeAbsent() {
  const rund::kernel::GraphBufferRef buffers[] = {
      {.logical_id = 11u, .role = rund::kernel::BufferRole::Read},
      {.logical_id = 21u, .role = rund::kernel::BufferRole::Write},
  };
  const rund::kernel::GraphNode node{
      .op_hash_hi = 0x1020304050607080u,
      .op_hash_lo = 0x8877665544332211u,
      .buffers = buffers,
      .buffer_count = 2u,
      .element_count = 1u,
  };
  const rund::kernel::Graph graph{
      .nodes = &node,
      .node_count = 1u,
      .scalar = rund::kernel::ComputeScalar::Lane32,
      .domain = rund::kernel::ComputeDomain::I32,
      .fixed_format = test::FixedFormatForLane(
          rund::kernel::ComputeScalar::Lane32),
  };
  const auto check = rund::kernel::ValidateGraph(graph);

  TEST_ASSERT(!check.ok);
  TEST_ASSERT(std::string_view{check.reason} ==
              "compute_graph_numeric_invalid");
  return 0;
}

} // namespace

int GraphRejectNumericPolicy() {
  if (UnsupportedScalar() != 0) {
    return 1;
  }
  return NonFixedFormatMustBeAbsent();
}

} // namespace program_compute_contract

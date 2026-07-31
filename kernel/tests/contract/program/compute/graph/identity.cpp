#include "contract/program/compute/graph/local.hpp"
#include "test/compute/fixed.hpp"

#include <string_view>

namespace program_compute_contract {

namespace {

int test_compute_graph_identity_is_stable() {
  const auto graph_a = MakeTwoMapGraph();
  const auto graph_b = MakeTwoMapGraph();

  TEST_ASSERT(graph_a.ok);
  TEST_ASSERT(graph_b.ok);
  TEST_ASSERT(graph_a.graph_id_hi == graph_b.graph_id_hi);
  TEST_ASSERT(graph_a.graph_id_lo == graph_b.graph_id_lo);
  TEST_ASSERT(graph_a.graph_id_hi == 0xf20519f7ca65f06eull);
  TEST_ASSERT(graph_a.graph_id_lo == 0xa2881dd1b5722914ull);
  return 0;
}

int test_compute_graph_identity_includes_node_order() {
  const auto graph_a = MakeTwoMapGraph();
  const auto graph_b =
      MakeTwoMapGraph(TwoMapGraphOptions{.reorder_nodes = true});

  TEST_ASSERT(graph_a.ok);
  TEST_ASSERT(graph_b.ok);
  TEST_ASSERT(GraphIdsDiffer(graph_a, graph_b));
  return 0;
}

int test_compute_graph_identity_includes_op_hash() {
  const auto graph_a = MakeTwoMapGraph();
  const auto graph_b = MakeTwoMapGraph(
      TwoMapGraphOptions{.first_op_hash_lo = 0x8877665544332212u});
  const auto graph_c = MakeTwoMapGraph(
      TwoMapGraphOptions{.first_op_hash_hi = 0x1020304050607081u});

  TEST_ASSERT(graph_a.ok);
  TEST_ASSERT(graph_b.ok);
  TEST_ASSERT(graph_c.ok);
  TEST_ASSERT(GraphIdsDiffer(graph_a, graph_b));
  TEST_ASSERT(GraphIdsDiffer(graph_a, graph_c));
  return 0;
}

int test_compute_graph_identity_includes_logical_buffer_id() {
  const auto graph_a = MakeTwoMapGraph();
  const auto graph_b =
      MakeTwoMapGraph(TwoMapGraphOptions{.first_read_id = 12u});

  TEST_ASSERT(graph_a.ok);
  TEST_ASSERT(graph_b.ok);
  TEST_ASSERT(GraphIdsDiffer(graph_a, graph_b));
  return 0;
}

int test_compute_graph_identity_includes_buffer_role() {
  const auto graph_a = MakeTwoMapGraph();
  const auto graph_b = MakeTwoMapGraph(
      TwoMapGraphOptions{.first_read_role = rund::kernel::BufferRole::Write});

  TEST_ASSERT(graph_a.ok);
  TEST_ASSERT(graph_b.ok);
  TEST_ASSERT(GraphIdsDiffer(graph_a, graph_b));
  return 0;
}

int test_compute_graph_identity_includes_buffer_init() {
  const rund::kernel::GraphBufferRef preserve[] = {
      {.logical_id = 11u, .role = rund::kernel::BufferRole::Read},
      {.logical_id = 21u, .role = rund::kernel::BufferRole::Write},
  };
  const rund::kernel::GraphBufferRef zero[] = {
      {.logical_id = 11u, .role = rund::kernel::BufferRole::Read},
      {.logical_id = 21u,
       .role = rund::kernel::BufferRole::Write,
       .init = rund::kernel::BufferInit::Zero},
  };
  rund::kernel::GraphNode node{
      .op_hash_hi = 0x1020304050607080u,
      .op_hash_lo = 0x8877665544332211u,
      .buffers = preserve,
      .buffer_count = 2u,
      .element_count = 1u,
  };
  const rund::kernel::u64 output = 21u;
  const rund::kernel::Graph graph{
      .nodes = &node,
      .node_count = 1u,
      .outputs = &output,
      .output_count = 1u,
      .scalar = rund::kernel::ComputeScalar::Lane32,
      .domain = rund::kernel::ComputeDomain::U32,
  };
  const auto first = rund::kernel::ValidateGraph(graph);
  node.buffers = zero;
  const auto second = rund::kernel::ValidateGraph(graph);
  TEST_ASSERT(first.ok);
  TEST_ASSERT(second.ok);
  TEST_ASSERT(GraphIdsDiffer(first, second));
  return 0;
}

int test_compute_graph_identity_includes_buffer_order() {
  const auto graph_a = MakeTwoMapGraph();
  const auto graph_b =
      MakeTwoMapGraph(TwoMapGraphOptions{.reorder_first_buffers = true});

  TEST_ASSERT(graph_a.ok);
  TEST_ASSERT(graph_b.ok);
  TEST_ASSERT(GraphIdsDiffer(graph_a, graph_b));
  return 0;
}

int test_compute_graph_identity_includes_output_order() {
  const rund::kernel::GraphBufferRef buffers[] = {
      {.logical_id = 11u, .role = rund::kernel::BufferRole::Read},
      {.logical_id = 21u, .role = rund::kernel::BufferRole::Write},
      {.logical_id = 31u, .role = rund::kernel::BufferRole::Write},
  };
  const rund::kernel::GraphNode node{
      .op_hash_hi = 0x1020304050607080u,
      .op_hash_lo = 0x8877665544332211u,
      .buffers = buffers,
      .buffer_count = 3u,
      .element_count = 1024u,
  };
  const rund::kernel::u64 ordered[] = {21u, 31u};
  const rund::kernel::u64 reversed[] = {31u, 21u};
  const auto first = rund::kernel::ValidateGraph(rund::kernel::Graph{
      .nodes = &node,
      .node_count = 1u,
      .outputs = ordered,
      .output_count = 2u,
      .scalar = rund::kernel::ComputeScalar::Lane32,
      .domain = rund::kernel::ComputeDomain::Fixed,
      .fixed_format =
          test::FixedFormatForLane(rund::kernel::ComputeScalar::Lane32),
  });
  const auto second = rund::kernel::ValidateGraph(rund::kernel::Graph{
      .nodes = &node,
      .node_count = 1u,
      .outputs = reversed,
      .output_count = 2u,
      .scalar = rund::kernel::ComputeScalar::Lane32,
      .domain = rund::kernel::ComputeDomain::Fixed,
      .fixed_format =
          test::FixedFormatForLane(rund::kernel::ComputeScalar::Lane32),
  });
  TEST_ASSERT(first.ok);
  TEST_ASSERT(second.ok);
  TEST_ASSERT(GraphIdsDiffer(first, second));
  return 0;
}

int test_compute_graph_identity_includes_scalar() {
  const auto graph_a = MakeTwoMapGraph();
  const auto graph_b = MakeTwoMapGraph(
      TwoMapGraphOptions{.scalar = rund::kernel::ComputeScalar::Lane64,
                         .domain = rund::kernel::ComputeDomain::Fixed});

  TEST_ASSERT(graph_a.ok);
  TEST_ASSERT(graph_b.ok);
  TEST_ASSERT(GraphIdsDiffer(graph_a, graph_b));
  return 0;
}

int test_compute_graph_identity_includes_fixed_numeric_policy() {
  const auto baseline = MakeTwoMapGraph();
  auto format = test::FixedFormatForLane(rund::kernel::ComputeScalar::Lane32);

  format.integer_bits = 16u;
  format.fraction_bits = 16u;
  const auto integer_fraction =
      MakeTwoMapGraph(TwoMapGraphOptions{.fixed_format = format});

  format = test::FixedFormatForLane(rund::kernel::ComputeScalar::Lane32);
  format.rounding = rund::kernel::ComputeRounding::Down;
  const auto rounding =
      MakeTwoMapGraph(TwoMapGraphOptions{.fixed_format = format});

  format = test::FixedFormatForLane(rund::kernel::ComputeScalar::Lane32);
  format.overflow = rund::kernel::ComputeOverflow::Wrap;
  const auto overflow =
      MakeTwoMapGraph(TwoMapGraphOptions{.fixed_format = format});

  format = test::FixedFormatForLane(rund::kernel::ComputeScalar::Lane32);
  format.approximation = rund::kernel::ComputeApproximation::Deterministic;
  const auto approximation =
      MakeTwoMapGraph(TwoMapGraphOptions{.fixed_format = format});

  TEST_ASSERT(baseline.ok);
  TEST_ASSERT(integer_fraction.ok);
  TEST_ASSERT(rounding.ok);
  TEST_ASSERT(overflow.ok);
  TEST_ASSERT(approximation.ok);
  TEST_ASSERT(GraphIdsDiffer(baseline, integer_fraction));
  TEST_ASSERT(GraphIdsDiffer(baseline, rounding));
  TEST_ASSERT(GraphIdsDiffer(baseline, overflow));
  TEST_ASSERT(GraphIdsDiffer(baseline, approximation));
  return 0;
}

int test_compute_graph_identity_includes_map_element_count() {
  const auto graph_a = MakeTwoMapGraph();
  const auto graph_b =
      MakeTwoMapGraph(TwoMapGraphOptions{.element_count = 1025u});

  TEST_ASSERT(graph_a.ok);
  TEST_ASSERT(graph_b.ok);
  TEST_ASSERT(GraphIdsDiffer(graph_a, graph_b));
  return 0;
}

int test_compute_graph_identity_includes_collective_kind() {
  const rund::kernel::GraphBufferRef buffers[] = {
      {.logical_id = 11u, .role = rund::kernel::BufferRole::Read},
      {.logical_id = 21u, .role = rund::kernel::BufferRole::Write},
  };
  rund::kernel::GraphNode node{
      .buffers = buffers,
      .buffer_count = 2u,
      .kind = rund::kernel::NodeKind::Scan,
      .primitive_hash_hi = 0x1111222233334444u,
      .primitive_hash_lo = 0x5555666677778888u,
      .element_count = 1024u,
  };
  const rund::kernel::Graph scan_graph{
      .nodes = &node,
      .node_count = 1u,
      .scalar = rund::kernel::ComputeScalar::Lane32,
  };
  const auto scan_check = rund::kernel::ValidateGraph(scan_graph);
  node.kind = rund::kernel::NodeKind::Sort;
  const rund::kernel::Graph sort_graph{
      .nodes = &node,
      .node_count = 1u,
      .scalar = rund::kernel::ComputeScalar::Lane32,
  };
  const auto sort_check = rund::kernel::ValidateGraph(sort_graph);

  TEST_ASSERT(scan_check.ok);
  TEST_ASSERT(sort_check.ok);
  TEST_ASSERT(GraphIdsDiffer(scan_check, sort_check));
  return 0;
}

int test_compute_graph_identity_includes_collective_descriptor() {
  const rund::kernel::GraphBufferRef buffers[] = {
      {.logical_id = 11u, .role = rund::kernel::BufferRole::Read},
      {.logical_id = 21u, .role = rund::kernel::BufferRole::Write},
  };
  rund::kernel::GraphNode node{
      .buffers = buffers,
      .buffer_count = 2u,
      .kind = rund::kernel::NodeKind::Compact,
      .primitive_hash_hi = 0x1111222233334444u,
      .primitive_hash_lo = 0x5555666677778888u,
      .element_count = 1024u,
  };
  const rund::kernel::Graph first_graph{
      .nodes = &node,
      .node_count = 1u,
      .scalar = rund::kernel::ComputeScalar::Lane32,
  };
  const auto first = rund::kernel::ValidateGraph(first_graph);
  node.primitive_hash_lo ^= 1u;
  const auto changed_hash = rund::kernel::ValidateGraph(first_graph);
  node.primitive_hash_lo ^= 1u;
  node.element_count += 1u;
  const auto changed_count = rund::kernel::ValidateGraph(first_graph);

  TEST_ASSERT(first.ok);
  TEST_ASSERT(changed_hash.ok);
  TEST_ASSERT(changed_count.ok);
  TEST_ASSERT(GraphIdsDiffer(first, changed_hash));
  TEST_ASSERT(GraphIdsDiffer(first, changed_count));
  return 0;
}

int test_compute_graph_identity_extends_only_active_control() {
  const rund::kernel::GraphBufferRef buffers[] = {
      {.logical_id = 11u, .role = rund::kernel::BufferRole::Read},
      {.logical_id = 12u, .role = rund::kernel::BufferRole::Read},
      {.logical_id = 21u, .role = rund::kernel::BufferRole::Write},
  };
  rund::kernel::GraphNode node{
      .op_hash_hi = 0x1020304050607080u,
      .op_hash_lo = 0x8877665544332211u,
      .buffers = buffers,
      .buffer_count = 3u,
      .element_count = 1024u,
  };
  const rund::kernel::Graph graph{
      .nodes = &node,
      .node_count = 1u,
      .scalar = rund::kernel::ComputeScalar::Lane32,
  };
  const auto descriptor = rund::kernel::ValidateGraph(graph);
  node.control = rund::kernel::GraphControl{
      .count_source = rund::kernel::GraphControlSource::U32,
      .count_binding = 1u,
      .capacity = node.element_count,
  };
  const auto counted_u32 = rund::kernel::ValidateGraph(graph);
  node.control.count_source = rund::kernel::GraphControlSource::U64;
  const auto counted_u64 = rund::kernel::ValidateGraph(graph);
  node.control.count_source = rund::kernel::GraphControlSource::U32;
  node.control.iteration = 2u;
  const auto counted_iteration = rund::kernel::ValidateGraph(graph);
  node.control.predicate_source = rund::kernel::GraphControlSource::U32;
  node.control.predicate_binding = 1u;
  node.control.predicate_expected = 1u;
  node.control.iteration = 3u;
  const auto predicated = rund::kernel::ValidateGraph(graph);

  TEST_ASSERT(descriptor.ok);
  TEST_ASSERT(counted_u32.ok);
  TEST_ASSERT(counted_u64.ok);
  TEST_ASSERT(counted_iteration.ok);
  TEST_ASSERT(predicated.ok);
  TEST_ASSERT(GraphIdsDiffer(descriptor, counted_u32));
  TEST_ASSERT(GraphIdsDiffer(counted_u32, counted_u64));
  TEST_ASSERT(GraphIdsDiffer(counted_u32, counted_iteration));
  TEST_ASSERT(GraphIdsDiffer(counted_u32, predicated));
  return 0;
}

int test_compute_zero_work_identity_uses_numeric_domain() {
  const rund::kernel::GraphBufferRef buffers[] = {
      {.logical_id = 1u, .role = rund::kernel::BufferRole::Read},
      {.logical_id = 2u, .role = rund::kernel::BufferRole::Write},
  };
  const rund::kernel::GraphNode node{
      .op_hash_hi = 0x1020304050607080u,
      .op_hash_lo = 0x8877665544332211u,
      .buffers = buffers,
      .buffer_count = 2u,
      .kind = rund::kernel::NodeKind::Map,
      .element_count = 0u,
  };
  const rund::kernel::u64 output = 2u;
  const rund::kernel::Graph graph{
      .nodes = &node,
      .node_count = 1u,
      .outputs = &output,
      .output_count = 1u,
      .scalar = rund::kernel::ComputeScalar::Lane32,
      .domain = rund::kernel::ComputeDomain::U32,
  };
  const auto first = rund::kernel::ValidateGraphIdentity(graph);
  const auto second = rund::kernel::ValidateGraphIdentity(graph);
  const auto execution = rund::kernel::ValidateGraph(graph);
  rund::kernel::Graph changed = graph;
  changed.domain = rund::kernel::ComputeDomain::I32;
  const auto changed_domain = rund::kernel::ValidateGraphIdentity(changed);
  TEST_ASSERT(first.ok);
  TEST_ASSERT(second.ok);
  TEST_ASSERT(first.graph_id_hi == second.graph_id_hi);
  TEST_ASSERT(first.graph_id_lo == second.graph_id_lo);
  TEST_ASSERT(changed_domain.ok);
  TEST_ASSERT(GraphIdsDiffer(first, changed_domain));
  TEST_ASSERT(!execution.ok);
  TEST_ASSERT(std::string_view{execution.reason} ==
              "compute_graph_primitive_invalid");
  return 0;
}

} // namespace

int RunGraphIdentityContract() {
  if (test_compute_graph_identity_is_stable() != 0) {
    return 1;
  }
  if (test_compute_graph_identity_includes_node_order() != 0) {
    return 1;
  }
  if (test_compute_graph_identity_includes_op_hash() != 0) {
    return 1;
  }
  if (test_compute_graph_identity_includes_logical_buffer_id() != 0) {
    return 1;
  }
  if (test_compute_graph_identity_includes_buffer_role() != 0) {
    return 1;
  }
  if (test_compute_graph_identity_includes_buffer_init() != 0) {
    return 1;
  }
  if (test_compute_graph_identity_includes_buffer_order() != 0) {
    return 1;
  }
  if (test_compute_graph_identity_includes_output_order() != 0) {
    return 1;
  }
  if (test_compute_graph_identity_includes_scalar() != 0) {
    return 1;
  }
  if (test_compute_graph_identity_includes_fixed_numeric_policy() != 0) {
    return 1;
  }
  if (test_compute_graph_identity_includes_map_element_count() != 0) {
    return 1;
  }
  if (test_compute_graph_identity_includes_collective_kind() != 0) {
    return 1;
  }
  if (test_compute_graph_identity_includes_collective_descriptor() != 0) {
    return 1;
  }
  if (test_compute_graph_identity_extends_only_active_control() != 0) {
    return 1;
  }
  return test_compute_zero_work_identity_uses_numeric_domain();
}

} // namespace program_compute_contract

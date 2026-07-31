#include "test/compute/fixed.hpp"
#include "contract/program/compute/graph/local.hpp"

#include <kernel/program/compute/factor/identity.hpp>
#include <kernel/program/compute/factor/plan.hpp>
#include <kernel/program/compute/factor/reference.hpp>
#include <kernel/program/compute/factor/model.hpp>
#include <kernel/program/compute/matrix/model.hpp>
#include <kernel/program/compute/matrix/identity.hpp>
#include <kernel/program/compute/matrix/plan.hpp>
#include <kernel/program/compute/matrix/reference.hpp>
#include <kernel/program/compute/partition/model.hpp>
#include <kernel/program/compute/partition/identity.hpp>
#include <kernel/program/compute/partition/plan.hpp>
#include <kernel/program/compute/partition/reference.hpp>
#include <kernel/program/compute/segmented/reduce/identity.hpp>
#include <kernel/program/compute/segmented/reduce/plan.hpp>
#include <kernel/program/compute/segmented/reduce/reference.hpp>
#include <kernel/program/compute/segmented/reduce/model.hpp>
#include <kernel/program/compute/segmented/scan/identity.hpp>
#include <kernel/program/compute/segmented/scan/plan.hpp>
#include <kernel/program/compute/segmented/scan/reference.hpp>
#include <kernel/program/compute/segmented/scan/model.hpp>
#include <kernel/program/compute/solve/identity.hpp>
#include <kernel/program/compute/solve/plan.hpp>
#include <kernel/program/compute/solve/reference.hpp>
#include <kernel/program/compute/solve/model.hpp>
#include <kernel/program/compute/spectrum/identity.hpp>
#include <kernel/program/compute/spectrum/plan.hpp>
#include <kernel/program/compute/spectrum/reference.hpp>
#include <kernel/program/compute/spectrum/model.hpp>
#include <kernel/program/compute/transform/model.hpp>
#include <kernel/program/compute/transform/identity.hpp>
#include <kernel/program/compute/transform/plan.hpp>
#include <kernel/program/compute/transform/reference.hpp>

namespace program_compute_contract {

namespace {

int test_compute_graph_admits_sort_collective_node() {
  const rund::kernel::GraphBufferRef map_buffers[] = {
      {.logical_id = 11u, .role = rund::kernel::BufferRole::Read},
      {.logical_id = 21u, .role = rund::kernel::BufferRole::Write},
  };
  const rund::kernel::GraphBufferRef sort_buffers[] = {
      {.logical_id = 21u, .role = rund::kernel::BufferRole::Read},
      {.logical_id = 31u, .role = rund::kernel::BufferRole::Write},
  };
  const rund::kernel::GraphNode nodes[] = {
      {.op_hash_hi = 0x1020304050607080u,
       .op_hash_lo = 0x8877665544332211u,
       .buffers = map_buffers,
       .buffer_count = 2u,
       .element_count = 256u},
      {.buffers = sort_buffers,
       .buffer_count = 2u,
       .kind = rund::kernel::NodeKind::Sort,
       .primitive_hash_hi = 0x3141592653589793u,
       .primitive_hash_lo = 0x2718281828459045u,
       .element_count = 256u},
  };
  const rund::kernel::Graph graph{
      .nodes = nodes,
      .node_count = 2u,
      .scalar = rund::kernel::ComputeScalar::Lane32,
  };
  const auto check = rund::kernel::ValidateGraph(graph);

  TEST_ASSERT(check.ok);
  TEST_ASSERT(check.node_count == 2u);
  return 0;
}

int test_compute_graph_admits_reduce_collective_node() {
  const rund::kernel::GraphBufferRef buffers[] = {
      {.logical_id = 11u, .role = rund::kernel::BufferRole::Read},
      {.logical_id = 21u, .role = rund::kernel::BufferRole::Write},
  };
  const rund::kernel::GraphNode node{
      .buffers = buffers,
      .buffer_count = 2u,
      .kind = rund::kernel::NodeKind::Reduce,
      .primitive_hash_hi = 0x1020304050607080u,
      .primitive_hash_lo = 0x8877665544332211u,
      .element_count = 512u,
  };
  const rund::kernel::Graph graph{
      .nodes = &node,
      .node_count = 1u,
      .scalar = rund::kernel::ComputeScalar::Lane32,
  };
  const auto check = rund::kernel::ValidateGraph(graph);

  TEST_ASSERT(check.ok);
  TEST_ASSERT(check.node_count == 1u);
  return 0;
}

int test_compute_graph_admits_partition_collective_node() {
  const rund::kernel::GraphBufferRef buffers[] = {
      {.logical_id = 11u, .role = rund::kernel::BufferRole::Read},
      {.logical_id = 12u, .role = rund::kernel::BufferRole::Read},
      {.logical_id = 21u, .role = rund::kernel::BufferRole::Write},
  };
  const rund::kernel::PartitionDesc desc{
      .element_count = 512u,
      .flag_bytes = 4u,
      .value_bytes = 4u,
  };
  const rund::kernel::PartitionHash hash = rund::kernel::HashPartition(desc);
  const rund::kernel::GraphNode node{
      .buffers = buffers,
      .buffer_count = 3u,
      .kind = rund::kernel::NodeKind::Partition,
      .primitive_hash_hi = hash.hi,
      .primitive_hash_lo = hash.lo,
      .element_count = desc.element_count,
  };
  const rund::kernel::Graph graph{
      .nodes = &node,
      .node_count = 1u,
      .scalar = rund::kernel::ComputeScalar::Lane32,
  };
  const auto check = rund::kernel::ValidateGraph(graph);

  TEST_ASSERT(check.ok);
  TEST_ASSERT(check.node_count == 1u);
  return 0;
}

int test_compute_graph_admits_segmented_scan_collective_node() {
  const rund::kernel::GraphBufferRef buffers[] = {
      {.logical_id = 11u, .role = rund::kernel::BufferRole::Read},
      {.logical_id = 12u, .role = rund::kernel::BufferRole::Read},
      {.logical_id = 21u, .role = rund::kernel::BufferRole::Write},
  };
  const rund::kernel::SegmentedScanDesc desc{
      .op = rund::kernel::SegmentedScanOp::InclusiveSum,
      .element = rund::kernel::SegmentedScanElement::U64,
      .element_count = 128u,
      .block_size = 32u,
  };
  const rund::kernel::SegmentedScanHash hash =
      rund::kernel::HashSegmentedScan(desc);
  rund::kernel::GraphNode node{
      .buffers = buffers,
      .buffer_count = 3u,
      .kind = rund::kernel::NodeKind::SegmentedScan,
      .primitive_hash_hi = hash.hi,
      .primitive_hash_lo = hash.lo,
      .element_count = desc.element_count,
  };
  const rund::kernel::Graph graph{
      .nodes = &node,
      .node_count = 1u,
      .scalar = rund::kernel::ComputeScalar::Lane64,
      .domain = rund::kernel::ComputeDomain::U64,
  };
  const auto first = rund::kernel::ValidateGraph(graph);
  node.primitive_hash_lo ^= 1u;
  const auto changed_descriptor = rund::kernel::ValidateGraph(graph);

  TEST_ASSERT(first.ok);
  TEST_ASSERT(changed_descriptor.ok);
  TEST_ASSERT(first.node_count == 1u);
  TEST_ASSERT(GraphIdsDiffer(first, changed_descriptor));
  return 0;
}

int test_compute_graph_admits_segmented_reduce_collective_node() {
  const rund::kernel::GraphBufferRef buffers[] = {
      {.logical_id = 11u, .role = rund::kernel::BufferRole::Read},
      {.logical_id = 12u, .role = rund::kernel::BufferRole::Read},
      {.logical_id = 21u, .role = rund::kernel::BufferRole::Write},
  };
  const rund::kernel::SegmentedReduceDesc desc{
      .op = rund::kernel::ReduceOp::Max,
      .element = rund::kernel::ReduceElement::U64,
      .element_count = 128u,
      .block_size = 32u,
  };
  const rund::kernel::SegmentedReduceHash hash =
      rund::kernel::HashSegmentedReduce(desc);
  const rund::kernel::GraphNode node{
      .buffers = buffers,
      .buffer_count = 3u,
      .kind = rund::kernel::NodeKind::SegmentedReduce,
      .primitive_hash_hi = hash.hi,
      .primitive_hash_lo = hash.lo,
      .element_count = desc.element_count,
  };
  const rund::kernel::Graph graph{
      .nodes = &node,
      .node_count = 1u,
      .scalar = rund::kernel::ComputeScalar::Lane64,
      .domain = rund::kernel::ComputeDomain::U64,
  };
  const auto check = rund::kernel::ValidateGraph(graph);

  TEST_ASSERT(check.ok);
  TEST_ASSERT(check.node_count == 1u);
  return 0;
}

int test_compute_graph_admits_transform_collective_node() {
  const rund::kernel::GraphBufferRef buffers[] = {
      {.logical_id = 11u, .role = rund::kernel::BufferRole::Read},
      {.logical_id = 12u, .role = rund::kernel::BufferRole::Read},
      {.logical_id = 21u, .role = rund::kernel::BufferRole::Write},
      {.logical_id = 22u, .role = rund::kernel::BufferRole::Write},
  };
  const rund::kernel::TransformDesc desc{
      .op = rund::kernel::TransformOp::Fourier,
      .direction = rund::kernel::TransformDir::Forward,
      .layout = rund::kernel::TransformLayout::Split,
      .normalization = rund::kernel::TransformNorm::None,
      .element_count = 128u,
  };
  const rund::kernel::TransformHash hash = rund::kernel::HashTransform(desc);
  const rund::kernel::GraphNode node{
      .buffers = buffers,
      .buffer_count = 4u,
      .kind = rund::kernel::NodeKind::Transform,
      .primitive_hash_hi = hash.hi,
      .primitive_hash_lo = hash.lo,
      .element_count = desc.element_count,
  };
  const rund::kernel::Graph graph{
      .nodes = &node,
      .node_count = 1u,
      .scalar = rund::kernel::ComputeScalar::Lane32,
  };
  const auto check = rund::kernel::ValidateGraph(graph);

  TEST_ASSERT(check.ok);
  TEST_ASSERT(check.node_count == 1u);
  return 0;
}

int test_compute_graph_admits_matrix_collective_node() {
  const rund::kernel::GraphBufferRef buffers[] = {
      {.logical_id = 11u, .role = rund::kernel::BufferRole::Read},
      {.logical_id = 12u, .role = rund::kernel::BufferRole::Read},
      {.logical_id = 21u, .role = rund::kernel::BufferRole::Write},
  };
  const rund::kernel::MatrixDesc desc{
      .op = rund::kernel::MatrixOp::Mul,
      .layout = rund::kernel::MatrixLayout::RowMajor,
      .rows = 2u,
      .cols = 3u,
      .inner = 4u,
      .batch_count = 1u,
      .element_bytes = sizeof(rund::kernel::u32),
      .fixed_format = test::FixedFormatForLane(
          rund::kernel::ComputeScalar::Lane32),
  };
  const rund::kernel::MatrixPlan plan = rund::kernel::PlanMatrix(desc);
  const rund::kernel::MatrixHash hash = rund::kernel::HashMatrix(desc);
  const rund::kernel::GraphNode node{
      .buffers = buffers,
      .buffer_count = 3u,
      .kind = rund::kernel::NodeKind::Matrix,
      .primitive_hash_hi = hash.hi,
      .primitive_hash_lo = hash.lo,
      .element_count = plan.output_count,
  };
  const rund::kernel::Graph graph{
      .nodes = &node,
      .node_count = 1u,
      .scalar = rund::kernel::ComputeScalar::Lane32,
  };
  const auto check = rund::kernel::ValidateGraph(graph);

  TEST_ASSERT(plan.ok);
  TEST_ASSERT(check.ok);
  TEST_ASSERT(check.node_count == 1u);
  return 0;
}

int test_compute_graph_admits_numeric_algebra_collective_nodes() {
  const rund::kernel::GraphBufferRef buffers[] = {
      {.logical_id = 11u, .role = rund::kernel::BufferRole::Read},
      {.logical_id = 21u, .role = rund::kernel::BufferRole::Write},
      {.logical_id = 31u, .role = rund::kernel::BufferRole::Write},
  };
  const rund::kernel::FactorDesc factor_desc{
      .op = rund::kernel::FactorOp::LU,
      .rows = 2u,
      .cols = 2u,
      .batch_count = 1u,
  };
  const rund::kernel::SolveDesc solve_desc{
      .op = rund::kernel::SolveOp::Linear,
      .input = rund::kernel::SolveInput::Matrix,
      .factor = rund::kernel::FactorOp::LU,
      .rows = 2u,
      .rhs_cols = 1u,
      .batch_count = 1u,
  };
  const rund::kernel::SpectrumDesc spectrum_desc{
      .op = rund::kernel::SpectrumOp::Eigen,
      .domain = rund::kernel::SpectrumDomain::SymmetricReal,
      .vectors = rund::kernel::SpectrumVectors::ValuesOnly,
      .rows = 2u,
      .cols = 2u,
      .batch_count = 1u,
      .max_iterations = 8u,
  };
  const rund::kernel::FactorHash factor_hash =
      rund::kernel::HashFactor(factor_desc);
  const rund::kernel::SolveHash solve_hash =
      rund::kernel::HashSolve(solve_desc);
  const rund::kernel::SpectrumHash spectrum_hash =
      rund::kernel::HashSpectrum(spectrum_desc);
  const rund::kernel::GraphNode nodes[] = {
      {.buffers = buffers,
       .buffer_count = 3u,
       .kind = rund::kernel::NodeKind::Factor,
       .primitive_hash_hi = factor_hash.hi,
       .primitive_hash_lo = factor_hash.lo,
       .element_count = 4u},
      {.buffers = buffers,
       .buffer_count = 3u,
       .kind = rund::kernel::NodeKind::Solve,
       .primitive_hash_hi = solve_hash.hi,
       .primitive_hash_lo = solve_hash.lo,
       .element_count = 2u},
      {.buffers = buffers,
       .buffer_count = 3u,
       .kind = rund::kernel::NodeKind::Spectrum,
       .primitive_hash_hi = spectrum_hash.hi,
       .primitive_hash_lo = spectrum_hash.lo,
       .element_count = 2u},
  };
  const rund::kernel::Graph graph{
      .nodes = nodes,
      .node_count = 3u,
      .scalar = rund::kernel::ComputeScalar::Lane32,
  };
  const auto check = rund::kernel::ValidateGraph(graph);

  TEST_ASSERT(check.ok);
  TEST_ASSERT(check.node_count == 3u);
  return 0;
}

} // namespace

int RunGraphAdmissionContract() {
  if (test_compute_graph_admits_sort_collective_node() != 0) {
    return 1;
  }
  if (test_compute_graph_admits_reduce_collective_node() != 0) {
    return 1;
  }
  if (test_compute_graph_admits_partition_collective_node() != 0) {
    return 1;
  }
  if (test_compute_graph_admits_segmented_scan_collective_node() != 0) {
    return 1;
  }
  if (test_compute_graph_admits_segmented_reduce_collective_node() != 0) {
    return 1;
  }
  if (test_compute_graph_admits_transform_collective_node() != 0) {
    return 1;
  }
  if (test_compute_graph_admits_matrix_collective_node() != 0) {
    return 1;
  }
  return test_compute_graph_admits_numeric_algebra_collective_nodes();
}

} // namespace program_compute_contract

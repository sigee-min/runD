#pragma once

#include <kernel/reduction/fold/graph/node.hpp>
#include <kernel/reduction/fold/strict.hpp>

#include <vector>

namespace rund::kernel {

struct FoldGraphProgram {
  FoldOperation operation = FoldOperation::FixedBinaryTreeHash;
  FoldValueDomain value_domain = FoldValueDomain::HashDigest;
  StrictFloatReductionPolicy strict_float_reduction{};
  u32 partition_count = 0u;
  u32 worker_local_slot_count = 0u;
  u32 global_ordered_slot_count = 0u;
  u32 scratch_slot_count = 0u;
  u32 final_slot = 0u;
  const FoldGraphNode *nodes = nullptr;
  u32 node_count = 0u;
  const FoldGraphEdge *edges = nullptr;
  u32 edge_count = 0u;
  bool fixed_topological_order = false;
  bool dag_validated = false;
  bool slot_bounds_validated = false;
  bool padding_law_validated = false;
  bool primitive_standardized = false;
  bool strict_floating_point = false;
  bool no_allocation = false;
};

struct FoldGraph {
  FoldOperation operation = FoldOperation::FixedBinaryTreeHash;
  FoldValueDomain value_domain = FoldValueDomain::HashDigest;
  StrictFloatReductionPolicy strict_float_reduction{};
  u32 partition_count = 0u;
  u32 worker_local_slot_count = 0u;
  u32 global_ordered_slot_count = 0u;
  u32 scratch_slot_count = 0u;
  u32 final_slot = 0u;
  bool fixed_binary_tree = true;
  bool fixed_topological_order = true;
  bool dag_validated = false;
  bool slot_bounds_validated = false;
  bool padding_law_validated = false;
  bool primitive_standardized = true;
  bool floating_point_allowed = false;
  bool strict_floating_point = false;
  std::vector<u32> partition_fold_slots{};
  std::vector<FoldGraphNode> nodes{};
  std::vector<FoldGraphEdge> reduction_edges{};
};

struct FoldGraphView {
  FoldOperation operation = FoldOperation::FixedBinaryTreeHash;
  FoldValueDomain value_domain = FoldValueDomain::HashDigest;
  StrictFloatReductionPolicy strict_float_reduction{};
  u32 partition_count = 0u;
  u32 worker_local_slot_count = 0u;
  u32 global_ordered_slot_count = 0u;
  u32 scratch_slot_count = 0u;
  u32 final_slot = 0u;
  bool fixed_binary_tree = true;
  bool fixed_topological_order = true;
  bool dag_validated = false;
  bool slot_bounds_validated = false;
  bool padding_law_validated = false;
  bool primitive_standardized = true;
  bool floating_point_allowed = false;
  bool strict_floating_point = false;
  const u32 *partition_fold_slots = nullptr;
  u32 partition_fold_slot_count = 0u;
  const FoldGraphNode *nodes = nullptr;
  u32 node_count = 0u;
  const FoldGraphEdge *reduction_edges = nullptr;
  u32 reduction_edge_count = 0u;
  FoldGraphProgram program{};
};

} // namespace rund::kernel

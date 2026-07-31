#pragma once

#include <kernel/reduction/fold/operation.hpp>

namespace rund::kernel {

struct FoldGraphBuild {
  bool ok = false;
  const char *reason = "not_run";
  u32 partition_count = 0u;
  u32 worker_local_slot_count = 0u;
  u32 global_ordered_slot_count = 0u;
  u32 scratch_slot_count = 0u;
  u32 node_count = 0u;
  u32 reduction_edge_count = 0u;
  u32 final_slot = 0u;
  FoldOperation operation = FoldOperation::FixedBinaryTreeHash;
  FoldValueDomain value_domain = FoldValueDomain::HashDigest;
  bool dag_validated = false;
  bool slot_bounds_validated = false;
  bool padding_law_validated = false;
  bool strict_floating_point = false;
  bool no_allocation = false;
};

struct FoldGraphValidationResult {
  bool ok = false;
  const char *reason = "not_run";
  u32 scratch_slot_count = 0u;
  u32 final_slot = 0u;
  bool dag_validated = false;
  bool slot_bounds_validated = false;
  bool padding_law_validated = false;
};

} // namespace rund::kernel

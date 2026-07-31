#pragma once

#include <kernel/program/compute/graph/validation.hpp>

namespace rund::kernel {

inline constexpr u64 kMaxFusionPolicyNodeCount = kMaxGraphNodeCount;
inline constexpr u64 kFusionBoundaryWordCount =
    (kMaxGraphNodeCount + 63u) / 64u;

struct FusionNodePolicy {
  u64 direct_read_mask = 0u;
  bool supported = false;
  bool writes_visible = false;
  u32 binding_count = 0u;
  u32 ir_node_count = 0u;
};

struct FusionPolicy {
  const FusionNodePolicy *nodes = nullptr;
  u64 node_count = 0u;
};

struct FusionPlan {
  u64 input_graph_id_hi = 0u;
  u64 input_graph_id_lo = 0u;
  u64 output_graph_id_hi = 0u;
  u64 output_graph_id_lo = 0u;
  u64 original_node_count = 0u;
  u64 fused_node_count = 0u;
  u64 rejected_edge_count = 0u;
  u64 fused_boundaries[kFusionBoundaryWordCount]{};
  bool ok = false;
  const char *reason = "compute_fusion_invalid";

  [[nodiscard]] constexpr explicit operator bool() const noexcept { return ok; }

  [[nodiscard]] constexpr bool boundary_fused(const u64 index) const noexcept {
    return original_node_count != 0u && index < original_node_count - 1u &&
           (fused_boundaries[index / 64u] & (u64{1u} << (index % 64u))) != 0u;
  }
};

[[nodiscard]] FusionPlan PlanFusion(const Graph &graph,
                                    const FusionPolicy &policy) noexcept;

} // namespace rund::kernel

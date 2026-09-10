#pragma once

#include <kernel/program/compute/fusion.hpp>

#include <limits>
#include <vector>

namespace rund::kernel::fusion_detail {

inline constexpr u64 kNoCandidate = std::numeric_limits<u64>::max();
inline constexpr u64 kBoundaryDecisionNone = 0u;
inline constexpr u64 kBoundaryDecisionFused = 1u;
inline constexpr u64 kBoundaryDecisionDependencyConflict = 2u;
inline constexpr u64 kBoundaryDecisionVisibilityBoundary = 3u;
inline constexpr u64 kBoundaryDecisionUnsupportedOp = 4u;
inline constexpr u64 kBoundaryDecisionCapacityBoundary = 5u;

struct BoundaryShape final {
  u64 producer_writes{};
  u64 consumer_reads{};
  u64 consumer_read_ordinal{};
  u64 intermediate{};
  bool candidate{};
};

struct BoundaryPlan final {
  u64 rejected_edges{};
  u64 intermediate{};
  u64 decision{kBoundaryDecisionNone};
  bool fused{};
  const char *reason{"compute_fusion_ok"};
};

struct ReaderFact final {
  u64 logical_id{};
  u64 writer{};
  u64 readers{};
  u64 active{kNoCandidate};
};

struct FusionHash final {
  u64 hi{};
  u64 lo{};
};

[[nodiscard]] u64 BuildReaderFacts(const Graph &, std::vector<BoundaryShape> &,
                                   std::vector<ReaderFact> &) noexcept;
[[nodiscard]] u64 ReaderCount(const ReaderFact *, u64 count, u64 logical_id,
                              u64 writer) noexcept;
[[nodiscard]] BoundaryPlan RejectBoundary(const char *reason, u64 decision,
                                          u64 intermediate) noexcept;
[[nodiscard]] BoundaryPlan EvaluateBoundary(const Graph &, const FusionPolicy &,
                                            const BoundaryShape &,
                                            const ReaderFact *, u64 fact_count,
                                            u64 left_index) noexcept;
[[nodiscard]] bool ValidPolicy(const FusionPolicy &, u64 node_count) noexcept;
[[nodiscard]] bool MergeFits(u64 binding_count, u64 ir_node_count,
                             const FusionNodePolicy &, u64 &merged_bindings,
                             u64 &merged_ir_nodes) noexcept;
[[nodiscard]] FusionHash Mix(FusionHash, u64 value) noexcept;
[[nodiscard]] FusionHash FusedOutputId(const GraphCheck &, const FusionPolicy &,
                                       u64 original_node_count,
                                       u64 fused_node_count,
                                       u64 rejected_edge_count,
                                       FusionHash boundary_decisions) noexcept;

} // namespace rund::kernel::fusion_detail

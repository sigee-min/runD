#include <kernel/program/compute/fusion.hpp>

#include <algorithm>
#include <cstddef>
#include <limits>
#include <new>
#include <vector>

namespace rund::kernel {
namespace {

inline constexpr u64 kNoCandidate = std::numeric_limits<u64>::max();
inline constexpr u64 kBoundaryDecisionNone = 0u;
inline constexpr u64 kBoundaryDecisionFused = 1u;
inline constexpr u64 kBoundaryDecisionDependencyConflict = 2u;
inline constexpr u64 kBoundaryDecisionVisibilityBoundary = 3u;
inline constexpr u64 kBoundaryDecisionUnsupportedOp = 4u;
inline constexpr u64 kBoundaryDecisionCapacityBoundary = 5u;

struct BoundaryShape {
  u64 producer_writes = 0u;
  u64 consumer_reads = 0u;
  u64 consumer_read_ordinal = 0u;
  u64 intermediate = 0u;
  bool candidate = false;
};

struct BoundaryPlan {
  u64 rejected_edges = 0u;
  u64 intermediate = 0u;
  u64 decision = kBoundaryDecisionNone;
  bool fused = false;
  const char *reason = "compute_fusion_ok";
};

struct ReaderFact {
  u64 logical_id = 0u;
  u64 writer = 0u;
  u64 readers = 0u;
  u64 active = kNoCandidate;
};

struct FusionHash {
  u64 hi = 0u;
  u64 lo = 0u;
};

[[nodiscard]] BoundaryShape InspectBoundary(const Graph &graph,
                                            const u64 left_index) noexcept {
  const GraphNode &left = graph.nodes[left_index];
  const GraphNode &right = graph.nodes[left_index + 1u];
  BoundaryShape shape{};
  for (u64 index = 0u; index < left.buffer_count; ++index) {
    const GraphBufferRef &buffer = left.buffers[index];
    if (buffer.role == BufferRole::Write) {
      ++shape.producer_writes;
      if (shape.producer_writes == 1u) {
        shape.intermediate = buffer.logical_id;
      }
    }
  }
  if (shape.producer_writes == 1u) {
    u64 read_ordinal = 0u;
    for (u64 index = 0u; index < right.buffer_count; ++index) {
      const GraphBufferRef &buffer = right.buffers[index];
      if (buffer.role == BufferRole::Read) {
        if (buffer.logical_id == shape.intermediate) {
          shape.candidate = true;
          shape.consumer_read_ordinal = read_ordinal;
          ++shape.consumer_reads;
        }
        ++read_ordinal;
      }
    }
    return shape;
  }

  // A multiple-write producer is never fusible, but it is still a rejected
  // candidate when any produced value crosses this boundary. Preserve the
  // first matching logical id as the deterministic rejection identity.
  for (u64 left_buffer = 0u; left_buffer < left.buffer_count; ++left_buffer) {
    const GraphBufferRef &producer = left.buffers[left_buffer];
    if (producer.role != BufferRole::Write) {
      continue;
    }
    for (u64 right_buffer = 0u; right_buffer < right.buffer_count;
         ++right_buffer) {
      const GraphBufferRef &consumer = right.buffers[right_buffer];
      if (consumer.role == BufferRole::Read &&
          consumer.logical_id == producer.logical_id) {
        shape.candidate = true;
        shape.consumer_reads = 1u;
        shape.intermediate = producer.logical_id;
        return shape;
      }
    }
  }
  return shape;
}

[[nodiscard]] u64 LowerBoundId(const ReaderFact *const facts, const u64 count,
                               const u64 logical_id) noexcept {
  u64 first = 0u;
  u64 length = count;
  while (length != 0u) {
    const u64 half = length / 2u;
    const u64 middle = first + half;
    if (facts[middle].logical_id < logical_id) {
      first = middle + 1u;
      length -= half + 1u;
    } else {
      length = half;
    }
  }
  return first;
}

[[nodiscard]] u64 UpperBoundId(const ReaderFact *const facts, const u64 count,
                               const u64 first, const u64 logical_id) noexcept {
  u64 begin = first;
  u64 length = count - first;
  while (length != 0u) {
    const u64 half = length / 2u;
    const u64 middle = begin + half;
    if (facts[middle].logical_id <= logical_id) {
      begin = middle + 1u;
      length -= half + 1u;
    } else {
      length = half;
    }
  }
  return begin;
}

[[nodiscard]] u64 FindWriter(const ReaderFact *const facts, const u64 first,
                             const u64 last, const u64 writer) noexcept {
  u64 begin = first;
  u64 length = last - first;
  while (length != 0u) {
    const u64 half = length / 2u;
    const u64 middle = begin + half;
    if (facts[middle].writer < writer) {
      begin = middle + 1u;
      length -= half + 1u;
    } else {
      length = half;
    }
  }
  return begin < last && facts[begin].writer == writer ? begin : kNoCandidate;
}

[[nodiscard]] u64 BuildReaderFacts(const Graph &graph,
                                   std::vector<BoundaryShape> &shapes,
                                   std::vector<ReaderFact> &facts) noexcept {
  u64 count = 0u;
  for (u64 boundary = 0u; boundary + 1u < graph.node_count; ++boundary) {
    BoundaryShape &shape = shapes[boundary];
    shape = InspectBoundary(graph, boundary);
    if (!shape.candidate || shape.producer_writes != 1u ||
        shape.consumer_reads != 1u) {
      continue;
    }
    facts[count++] =
        ReaderFact{.logical_id = shape.intermediate, .writer = boundary};
  }
  std::sort(facts.begin(), facts.begin() + static_cast<std::ptrdiff_t>(count),
            [](const ReaderFact &left, const ReaderFact &right) noexcept {
              return left.logical_id < right.logical_id ||
                     (left.logical_id == right.logical_id &&
                      left.writer < right.writer);
            });

  for (u64 node_index = 0u; node_index < graph.node_count; ++node_index) {
    const GraphNode &node = graph.nodes[node_index];
    for (u64 buffer_index = 0u; buffer_index < node.buffer_count;
         ++buffer_index) {
      const GraphBufferRef &buffer = node.buffers[buffer_index];
      if (buffer.role != BufferRole::Read) {
        continue;
      }
      const u64 first = LowerBoundId(facts.data(), count, buffer.logical_id);
      if (first < count && facts[first].logical_id == buffer.logical_id &&
          facts[first].active != kNoCandidate) {
        ++facts[facts[first].active].readers;
      }
    }
    for (u64 buffer_index = 0u; buffer_index < node.buffer_count;
         ++buffer_index) {
      const GraphBufferRef &buffer = node.buffers[buffer_index];
      if (buffer.role != BufferRole::Write) {
        continue;
      }
      const u64 first = LowerBoundId(facts.data(), count, buffer.logical_id);
      if (first == count || facts[first].logical_id != buffer.logical_id) {
        continue;
      }
      const u64 last =
          UpperBoundId(facts.data(), count, first, buffer.logical_id);
      facts[first].active = FindWriter(facts.data(), first, last, node_index);
    }
  }
  return count;
}

[[nodiscard]] u64 ReaderCount(const ReaderFact *const facts, const u64 count,
                              const u64 logical_id, const u64 writer) noexcept {
  const u64 first = LowerBoundId(facts, count, logical_id);
  if (first == count || facts[first].logical_id != logical_id) {
    return 0u;
  }
  const u64 last = UpperBoundId(facts, count, first, logical_id);
  const u64 found = FindWriter(facts, first, last, writer);
  return found == kNoCandidate ? 0u : facts[found].readers;
}

[[nodiscard]] BoundaryPlan RejectBoundary(const char *const reason,
                                          const u64 decision,
                                          const u64 intermediate) noexcept {
  return BoundaryPlan{.rejected_edges = 1u,
                      .intermediate = intermediate,
                      .decision = decision,
                      .reason = reason};
}

[[nodiscard]] BoundaryPlan
EvaluateBoundary(const Graph &graph, const FusionPolicy &policy,
                 const BoundaryShape &shape, const ReaderFact *const facts,
                 const u64 fact_count, const u64 left_index) noexcept {
  const GraphNode &left = graph.nodes[left_index];
  const GraphNode &right = graph.nodes[left_index + 1u];
  if (!shape.candidate) {
    return {};
  }
  if (policy.nodes[left_index].writes_visible) {
    return RejectBoundary("compute_fusion_visibility_boundary",
                          kBoundaryDecisionVisibilityBoundary,
                          shape.intermediate);
  }
  if (shape.producer_writes != 1u || shape.consumer_reads != 1u) {
    return RejectBoundary("compute_fusion_dependency_conflict",
                          kBoundaryDecisionDependencyConflict,
                          shape.intermediate);
  }
  if (ReaderCount(facts, fact_count, shape.intermediate, left_index) != 1u) {
    return RejectBoundary("compute_fusion_dependency_conflict",
                          kBoundaryDecisionDependencyConflict,
                          shape.intermediate);
  }
  if (left.kind != NodeKind::Map || right.kind != NodeKind::Map ||
      !policy.nodes[left_index].supported ||
      !policy.nodes[left_index + 1u].supported ||
      left.element_count != right.element_count) {
    return RejectBoundary("compute_fusion_unsupported_op",
                          kBoundaryDecisionUnsupportedOp, shape.intermediate);
  }
  if (shape.consumer_read_ordinal >= 64u ||
      (policy.nodes[left_index + 1u].direct_read_mask &
       (u64{1u} << shape.consumer_read_ordinal)) == 0u) {
    return RejectBoundary("compute_fusion_dependency_conflict",
                          kBoundaryDecisionDependencyConflict,
                          shape.intermediate);
  }
  return BoundaryPlan{.intermediate = shape.intermediate,
                      .decision = kBoundaryDecisionFused,
                      .fused = true};
}

[[nodiscard]] FusionHash Mix(const FusionHash hash, const u64 value) noexcept {
  const graph_detail::GraphHash mixed = graph_detail::Mix(
      graph_detail::GraphHash{.hi = hash.hi, .lo = hash.lo}, value);
  return FusionHash{.hi = mixed.hi, .lo = mixed.lo};
}

[[nodiscard]] FusionHash MixPolicy(FusionHash hash,
                                   const FusionPolicy &policy) noexcept {
  hash = Mix(hash, policy.node_count);
  for (u64 index = 0u; index < policy.node_count; ++index) {
    hash = Mix(hash, index);
    hash = Mix(hash, policy.nodes[index].direct_read_mask);
    hash = Mix(hash, policy.nodes[index].supported ? 1u : 0u);
    hash = Mix(hash, policy.nodes[index].writes_visible ? 1u : 0u);
    hash = Mix(hash, policy.nodes[index].binding_count);
    hash = Mix(hash, policy.nodes[index].ir_node_count);
  }
  return hash;
}

[[nodiscard]] FusionHash
FusedOutputId(const GraphCheck &input, const FusionPolicy &policy,
              const u64 original_node_count, const u64 fused_node_count,
              const u64 rejected_edge_count,
              const FusionHash boundary_decisions) noexcept {
  if (original_node_count == fused_node_count && rejected_edge_count == 0u) {
    return FusionHash{.hi = input.graph_id_hi, .lo = input.graph_id_lo};
  }
  FusionHash hash{.hi = input.graph_id_hi, .lo = input.graph_id_lo};
  hash = Mix(hash, 0x667573696f6e5f76ull);
  hash = MixPolicy(hash, policy);
  hash = Mix(hash, original_node_count);
  hash = Mix(hash, fused_node_count);
  hash = Mix(hash, rejected_edge_count);
  hash = Mix(hash, boundary_decisions.hi);
  hash = Mix(hash, boundary_decisions.lo);
  return hash;
}

[[nodiscard]] bool ValidPolicy(const FusionPolicy &policy,
                               const u64 node_count) noexcept {
  if (policy.node_count != node_count ||
      policy.node_count > kMaxFusionPolicyNodeCount ||
      (policy.node_count != 0u && policy.nodes == nullptr)) {
    return false;
  }
  for (u64 index = 0u; index < policy.node_count; ++index) {
    const FusionNodePolicy &node = policy.nodes[index];
    if (node.supported !=
        (node.binding_count != 0u && node.ir_node_count != 0u)) {
      return false;
    }
    if (node.binding_count > kMaxComputeBindingCount ||
        node.ir_node_count > kMaxComputeNodeCount) {
      return false;
    }
  }
  return true;
}

[[nodiscard]] bool MergeFits(const u64 binding_count, const u64 ir_node_count,
                             const FusionNodePolicy &right,
                             u64 &merged_bindings,
                             u64 &merged_ir_nodes) noexcept {
  merged_bindings = binding_count + right.binding_count - 2u;
  merged_ir_nodes = ir_node_count + right.ir_node_count - 2u;
  return merged_bindings != 0u && merged_ir_nodes != 0u &&
         merged_bindings <= kMaxComputeBindingCount &&
         merged_ir_nodes <= kMaxComputeNodeCount;
}

} // namespace

FusionPlan PlanFusion(const Graph &graph, const FusionPolicy &policy) noexcept {
  const GraphCheck input = ValidateGraph(graph);
  if (!input.ok) {
    return FusionPlan{.original_node_count = graph.node_count,
                      .reason = input.reason};
  }
  if (!ValidPolicy(policy, graph.node_count)) {
    return FusionPlan{.input_graph_id_hi = input.graph_id_hi,
                      .input_graph_id_lo = input.graph_id_lo,
                      .original_node_count = graph.node_count,
                      .fused_node_count = graph.node_count,
                      .reason = "compute_fusion_policy_invalid"};
  }

  std::vector<BoundaryShape> shapes;
  std::vector<ReaderFact> facts;
  try {
    shapes.resize(static_cast<std::size_t>(graph.node_count));
    facts.resize(static_cast<std::size_t>(graph.node_count));
  } catch (const std::bad_alloc &) {
    return FusionPlan{.input_graph_id_hi = input.graph_id_hi,
                      .input_graph_id_lo = input.graph_id_lo,
                      .original_node_count = graph.node_count,
                      .fused_node_count = graph.node_count,
                      .reason = "compute_fusion_capacity"};
  }
  const u64 fact_count = BuildReaderFacts(graph, shapes, facts);
  FusionPlan result{};
  u64 fused_boundary_count = 0u;
  u64 rejected_edge_count = 0u;
  const char *first_rejection = "compute_fusion_ok";
  FusionHash decisions{.hi = 0xc6a4a7935bd1e995ull,
                       .lo = 0x9e3779b97f4a7c15ull};
  u64 region_bindings = policy.nodes[0].binding_count;
  u64 region_ir_nodes = policy.nodes[0].ir_node_count;
  for (u64 boundary_index = 0u; boundary_index + 1u < graph.node_count;
       ++boundary_index) {
    BoundaryPlan boundary =
        EvaluateBoundary(graph, policy, shapes[boundary_index], facts.data(),
                         fact_count, boundary_index);
    if (boundary.fused) {
      u64 merged_bindings = 0u;
      u64 merged_ir_nodes = 0u;
      if (MergeFits(region_bindings, region_ir_nodes,
                    policy.nodes[boundary_index + 1u], merged_bindings,
                    merged_ir_nodes)) {
        region_bindings = merged_bindings;
        region_ir_nodes = merged_ir_nodes;
      } else {
        boundary = RejectBoundary("compute_fusion_capacity_boundary",
                                  kBoundaryDecisionCapacityBoundary,
                                  boundary.intermediate);
      }
    }
    if (!boundary.fused) {
      region_bindings = policy.nodes[boundary_index + 1u].binding_count;
      region_ir_nodes = policy.nodes[boundary_index + 1u].ir_node_count;
    }
    decisions = Mix(decisions, boundary_index);
    decisions = Mix(decisions, boundary.decision);
    decisions = Mix(decisions, boundary.intermediate);
    decisions = Mix(decisions, boundary.rejected_edges);
    decisions = Mix(decisions, boundary.fused ? 1u : 0u);
    if (boundary.rejected_edges != 0u && rejected_edge_count == 0u) {
      first_rejection = boundary.reason;
    }
    rejected_edge_count += boundary.rejected_edges;
    if (boundary.fused) {
      ++fused_boundary_count;
      result.fused_boundaries[boundary_index / 64u] |=
          u64{1u} << (boundary_index % 64u);
    }
  }

  const u64 fused_node_count = graph.node_count - fused_boundary_count;
  const FusionHash output =
      FusedOutputId(input, policy, graph.node_count, fused_node_count,
                    rejected_edge_count, decisions);
  result.input_graph_id_hi = input.graph_id_hi;
  result.input_graph_id_lo = input.graph_id_lo;
  result.output_graph_id_hi = output.hi;
  result.output_graph_id_lo = output.lo;
  result.original_node_count = graph.node_count;
  result.fused_node_count = fused_node_count;
  result.rejected_edge_count = rejected_edge_count;
  result.ok = true;
  result.reason =
      rejected_edge_count == 0u ? "compute_fusion_ok" : first_rejection;
  return result;
}

} // namespace rund::kernel

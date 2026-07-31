#include "fusion/local.hpp"

#include <kernel/program/compute/lowering/emission.hpp>
#include <kernel/program/compute/lowering/fusion/build.hpp>

#include <limits>
#include <span>
#include <utility>
#include <vector>

namespace rund::node::accel::detail {
namespace {

[[nodiscard]] std::uint64_t RunEnd(const rund::kernel::FusionPlan &fusion,
                                   const std::uint64_t first) noexcept {
  std::uint64_t end = first + 1u;
  while (end < fusion.original_node_count && fusion.boundary_fused(end - 1u)) {
    ++end;
  }
  return end;
}

[[nodiscard]] rund::kernel::Graph
RegionGraph(const rund::kernel::Graph &graph, const std::uint64_t first,
            const std::uint64_t count) noexcept {
  return rund::kernel::Graph{
      .nodes = graph.nodes + first,
      .node_count = count,
      .scalar = graph.scalar,
      .domain = graph.domain,
      .fixed_format = graph.fixed_format,
  };
}

[[nodiscard]] bool
AppendRemovedDispatches(FinalGraphSteps &result,
                        const FrozenDispatchCount original,
                        const FrozenDispatchCount fused) noexcept {
  if (!original.ok || !fused.ok || original.count <= fused.count) {
    return false;
  }
  const std::uint64_t removed = original.count - fused.count;
  if (result.removed_dispatch_count >
      std::numeric_limits<std::uint64_t>::max() - removed) {
    return false;
  }
  result.removed_dispatch_count += removed;
  return true;
}

[[nodiscard]] bool AppendOriginalStep(GraphCompileNode &node,
                                      const std::uint64_t source,
                                      FinalGraphSteps &result) {
  if (source >= static_cast<std::uint64_t>(SourceStep::none) - 1u) {
    return false;
  }
  KernelExecutionStep step = BuildKernelExecutionStep(std::move(node));
  if (step.kind() == rund::kernel::NodeKind::Map && !step.artifact.ok) {
    return false;
  }
  step.source =
      SourceRange{.begin = SourceStep{static_cast<std::uint32_t>(source)},
                  .end = SourceStep{static_cast<std::uint32_t>(source + 1u)}};
  result.steps.push_back(std::move(step));
  return true;
}

[[nodiscard]] bool AppendFusedRegion(
    const rund::kernel::Graph &graph, const rund::kernel::FusionPolicy &policy,
    GraphCompileState &state, const rund::kernel::ComputeCaps &caps,
    const std::uint64_t first, const std::uint64_t count,
    std::vector<const rund::kernel::ComputeIR *> &chain,
    std::vector<rund::kernel::compute_lowering_detail::ComputeInputAdmission *>
        &inputs,
    FinalGraphSteps &result) {
  chain.clear();
  inputs.clear();
  for (std::uint64_t offset = 0u; offset < count; ++offset) {
    GraphCompileNode &node = state.compile_nodes[first + offset];
    if (node.ir == nullptr) {
      return false;
    }
    chain.push_back(node.ir);
    inputs.push_back(&node.cpu_input);
  }

  const rund::kernel::Graph region = RegionGraph(graph, first, count);
  const rund::kernel::FusionPolicy region_policy{
      .nodes = policy.nodes + first,
      .node_count = count,
  };
  auto admitted = rund::kernel::compute_lowering_detail::
      BuildAdmittedFusedComputeMapChainIR(chain.data(), inputs.data(), count,
                                          region, region_policy, caps.api);
  rund::kernel::ComputeFusedMapChainIR &fused = admitted.value;
  if (!fused.ok || admitted.source_parse_count != count || !admitted.input.ok ||
      !fused.metadata.ok) {
    return false;
  }

  const std::span<const GraphCompileNode> nodes{
      state.compile_nodes.data() + first, static_cast<std::size_t>(count)};
  FusedStepBindings bindings =
      FusedStepBindingsFor(region, nodes, fused.metadata);
  if (!bindings.ok) {
    return false;
  }

  const FrozenDispatchCount original =
      BuildOriginalDispatchCount(nodes, caps, first);
  const FrozenDispatchCount final = BuildMapDispatchCount(
      fused.metadata, graph.nodes[first].element_count, caps, first + 1u);
  if (!AppendRemovedDispatches(result, original, final)) {
    return false;
  }

  auto emitted = rund::kernel::compute_lowering_detail::
      EmitGeneratedRetainedComputeArtifact(std::move(fused.ir),
                                           std::move(fused.metadata),
                                           std::move(admitted.input));
  if (!emitted.artifact.ok || !emitted.input.ok ||
      emitted.emission_count != 1u ||
      emitted.input.key != emitted.artifact.key ||
      !emitted.artifact.canonical_ir_bytes.empty()) {
    return false;
  }

  result.steps.push_back(BuildMapKernelExecutionStep(
      std::move(emitted.artifact), std::move(emitted.input),
      std::move(bindings.indices), graph.nodes[first].element_count));
  if (!result.steps.back().artifact.ok || first >= SourceStep::none ||
      count >= static_cast<std::uint64_t>(SourceStep::none) - first) {
    return false;
  }
  result.steps.back().source =
      SourceRange{.begin = SourceStep{static_cast<std::uint32_t>(first)},
                  .end = SourceStep{static_cast<std::uint32_t>(first + count)}};
  return true;
}

} // namespace

FinalGraphSteps BuildFinalGraphSteps(const rund::kernel::Graph &graph,
                                     const rund::kernel::FusionPlan &fusion,
                                     GraphCompileState &state,
                                     const rund::kernel::ComputeCaps &caps) {
  FinalGraphSteps result{
      .fused_operation_count = fusion.fused_node_count,
      .fusion_rejection_count = fusion.rejected_edge_count,
      .fusion_reason = fusion.reason,
      .ok = true,
      .reason = "ok",
  };
  if (graph.nodes == nullptr ||
      state.compile_nodes.size() != graph.node_count ||
      state.required_barriers.size() != graph.node_count ||
      fusion.original_node_count != graph.node_count ||
      fusion.fused_node_count == 0u ||
      fusion.fused_node_count > fusion.original_node_count) {
    return RejectFinalGraphSteps("accel_kernel_graph_invalid", fusion);
  }

  const rund::kernel::FusionPolicy policy = FusionPolicyFor(state.fusion_nodes);
  result.steps.reserve(static_cast<std::size_t>(fusion.fused_node_count));
  result.barriers.reserve(static_cast<std::size_t>(fusion.fused_node_count));
  std::vector<const rund::kernel::ComputeIR *> chain{};
  std::vector<rund::kernel::compute_lowering_detail::ComputeInputAdmission *>
      inputs{};
  chain.reserve(state.compile_nodes.size());
  inputs.reserve(state.compile_nodes.size());

  std::uint64_t first = 0u;
  while (first < graph.node_count) {
    const std::uint64_t end = RunEnd(fusion, first);
    const std::uint64_t count = end - first;
    const bool appended =
        count == 1u
            ? AppendOriginalStep(state.compile_nodes[first], first, result)
            : AppendFusedRegion(graph, policy, state, caps, first, count, chain,
                                inputs, result);
    if (!appended) {
      return RejectFinalGraphSteps("accel_kernel_artifact_invalid", fusion);
    }
    result.barriers.push_back(state.required_barriers[first]);
    first = end;
  }

  if (result.steps.size() != fusion.fused_node_count ||
      result.barriers.size() != result.steps.size() ||
      (fusion.fused_node_count < fusion.original_node_count) !=
          (result.removed_dispatch_count != 0u)) {
    return RejectFinalGraphSteps("accel_kernel_graph_invalid", fusion);
  }
  return result;
}

} // namespace rund::node::accel::detail

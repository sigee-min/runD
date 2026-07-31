#pragma once

#include <accel/context/value.hpp>
#include <accel/graph/value.hpp>
#include <accel/kernel/value.hpp>

#include "check.hpp"

#include "../compile.hpp"
#include "../fusion.hpp"
#include "../step.hpp"
#include "../token.hpp"

#include <kernel/program/compute/fusion.hpp>

#include <utility>
#include <vector>

namespace rund::node::accel {

rund::AccelKernel CompileAccelKernel(const rund::AccelContext &context,
                                     const rund::AccelGraph &graph) {
  const detail::ContextAdmission admission =
      detail::AdmitContextForSupport(context);
  if (!admission.check.ok) {
    return RejectKernel("accel_kernel_graph_invalid", admission);
  }
  if (!detail::GraphShapeCanBeWalked(graph) || graph.node_count == 0u) {
    return RejectKernel("accel_kernel_graph_invalid", admission);
  }

  detail::GraphCompileState state{};
  detail::ReserveGraphCompileState(state, graph.node_count);
  if (!detail::ReserveExplicitGraphLogicalIds(graph, state)) {
    return RejectKernel("accel_kernel_graph_invalid", admission);
  }
  for (std::uint64_t node_index = 0u; node_index < graph.node_count;
       ++node_index) {
    const char *const reason =
        detail::AppendGraphCompileNode(admission, graph, node_index, state);
    if (!ReasonOk(reason)) {
      return RejectKernel(reason, admission);
    }
  }

  const rund::kernel::Graph kernel_graph = detail::KernelGraphFor(
      state, graph.scalar, graph.domain, graph.fixed_format, graph.outputs,
      graph.output_count);
  const rund::kernel::GraphCheck graph_check =
      rund::kernel::ValidateGraph(kernel_graph);
  if (!graph_check.ok) {
    return RejectKernel("accel_kernel_graph_invalid", admission);
  }

  const rund::kernel::FusionPolicy fusion_policy =
      detail::FusionPolicyFor(state.fusion_nodes);
  const rund::kernel::FusionPlan fusion =
      rund::kernel::PlanFusion(kernel_graph, fusion_policy);
  if (!fusion.ok) {
    return RejectKernel(fusion.reason, admission);
  }

  detail::FinalGraphSteps final_steps =
      detail::BuildFinalGraphSteps(kernel_graph, fusion, state, admission.caps);
  if (!final_steps.ok) {
    return RejectKernel(final_steps.reason, admission);
  }

  rund::AccelKernel kernel{
      .check = OkGraphCheck(),
      .graph_id_hi = graph_check.graph_id_hi,
      .graph_id_lo = graph_check.graph_id_lo,
      .node_count = graph_check.node_count,
      .api = admission.api,
      .scalar = graph.scalar,
      .domain = graph.domain,
      .frozen_caps = admission.caps,
      .context_id = admission.context_id,
      .reason = "ok",
  };
  const detail::KernelTokenMint mint = detail::MintKernelToken(
      admission, graph_check, std::move(final_steps.steps),
      std::move(final_steps.barriers), final_steps.removed_dispatch_count,
      std::move(state.graph_roles), std::move(state.graph_shapes),
      std::move(state.graph_visibilities),
      std::move(state.graph_alias_representatives),
      std::move(state.graph_binding_sources),
      std::move(state.graph_reset_bindings), graph.scalar, graph.domain,
      fusion.original_node_count, final_steps.fused_operation_count,
      final_steps.fusion_rejection_count, final_steps.fusion_reason);
  if (mint.owner == nullptr || mint.kernel_id == 0u) {
    return RejectKernel("accel_kernel_graph_invalid", admission);
  }
  kernel.owner = mint.owner;
  kernel.kernel_id = mint.kernel_id;
  return kernel;
}
} // namespace rund::node::accel

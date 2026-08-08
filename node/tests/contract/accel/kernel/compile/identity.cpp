#include <accel/buffer.hpp>
#include <accel/context/buffer.hpp>
#include <accel/graph/buffer/ref.hpp>
#include <accel/graph/node.hpp>
#include <accel/graph/value.hpp>
#include <accel/kernel/value.hpp>

#include "local.hpp"
#include <node/accel/context.hpp>

#include "src/accel/graph/token/local.hpp"

#include <kernel/program/compute/lowering/artifact/admission.hpp>
#include <kernel/program/compute/plan.hpp>

#include <memory>
#include <type_traits>

namespace node_accel_contract::kernel_case::compile {

bool IdentityIsStable(const Fixture &fixture) {
  using AdmitToken =
      std::shared_ptr<rund::node::accel::detail::KernelToken> (*)(
          const rund::AccelKernel &,
          const rund::node::accel::detail::ContextAdmission &);
  static_assert(
      std::is_same_v<
          decltype(&rund::node::accel::detail::AdmitKernelTokenWithContext),
          AdmitToken>);

  const rund::AccelKernel &first = fixture.first;
  const rund::AccelKernel &second = fixture.second;
  if (!first.check.ok || !second.check.ok || first.kernel_id == 0u ||
      second.kernel_id == 0u || first.graph_id_hi == 0u ||
      first.graph_id_lo == 0u || first.graph_id_hi != second.graph_id_hi ||
      first.graph_id_lo != second.graph_id_lo ||
      first.api != fixture.context.api ||
      first.scalar != rund::kernel::ComputeScalar::Lane32 ||
      first.node_count != fixture.graph.node_count || !first.frozen_caps.ok ||
      first.context_id != fixture.context.id || first.owner == nullptr ||
      rund::node::test::SameOwner(first.owner, fixture.context.owner)) {
    return false;
  }

  const rund::node::accel::detail::ContextAdmission context_admission =
      rund::node::accel::detail::AdmitContextForSupport(fixture.context);
  const std::shared_ptr<rund::node::accel::detail::KernelToken> token =
      rund::node::accel::detail::AdmitKernelTokenWithContext(first,
                                                             context_admission);
  const std::shared_ptr<void> token_owner =
      std::static_pointer_cast<void>(token);
  if (token == nullptr || token.get() != first.owner.get() ||
      !rund::node::test::SameOwner(token_owner, first.owner) ||
      token->kernel_id != first.kernel_id ||
      token->context_id != first.context_id ||
      token->graph_id_hi != first.graph_id_hi ||
      token->graph_id_lo != first.graph_id_lo ||
      token->node_count != first.node_count || token->api != first.api ||
      token->scalar != first.scalar || token->domain != first.domain) {
    return false;
  }
  rund::AccelKernel tampered = first;
  ++tampered.node_count;
  if (rund::node::accel::detail::AdmitKernelTokenWithContext(
          tampered, context_admission) != nullptr) {
    return false;
  }

  const rund::node::accel::detail::KernelAdmission admitted =
      rund::node::accel::detail::AdmitKernelForSupport(fixture.context, first);
  if (!admitted.check.ok || admitted.graph_id_hi != first.graph_id_hi ||
      admitted.graph_id_lo != first.graph_id_lo ||
      admitted.context_id != fixture.context.id || admitted.api != first.api ||
      admitted.scalar != first.scalar) {
    return false;
  }

  const rund::node::accel::detail::KernelExecution execution =
      rund::node::accel::detail::AdmitKernelForExecution(fixture.context,
                                                         first);
  const rund::node::accel::detail::KernelExecution repeated =
      rund::node::accel::detail::AdmitKernelForExecution(fixture.context,
                                                         first);
  if (!execution.admission.check.ok || execution.steps.size() != 1u) {
    return false;
  }
  const auto &step = execution.steps.front();
  const rund::kernel::ComputePlan plan = rund::kernel::PlanCompute(
      rund::kernel::TilePhaseDescription{
          .phase_id = first.kernel_id,
          .tile_count = step.element_count,
          .capacity =
              rund::kernel::TilePhaseCapacityRequirement{
                  .scratch_alignment = 1u,
                  .output_shards = step.element_count,
                  .queue_slots = step.element_count,
                  .task_slots = step.element_count,
              },
      },
      step.artifact.metadata.map, first.frozen_caps,
      rund::kernel::ComputeLimit{
          .staging_bytes = first.frozen_caps.staging_bytes,
          .max_window_tiles = first.frozen_caps.max_window_tiles,
      });
  const auto warm = rund::kernel::compute_lowering_detail::AdmitRetained(
      plan, step.artifact, nullptr);
  return execution.admission.check.ok && execution.steps.size() == 1u &&
         repeated.admission.check.ok &&
         repeated.steps.size() == execution.steps.size() &&
         repeated.steps.data() == execution.steps.data() &&
         execution.steps.front().artifact.ok &&
         execution.steps.front().artifact.key.api == first.frozen_caps.api &&
         execution.steps.front().artifact.key.scalar == first.scalar &&
         execution.steps.front().artifact.key.op_hash_hi ==
             fixture.op.ir().op_hash_hi &&
         execution.steps.front().artifact.key.op_hash_lo ==
             fixture.op.ir().op_hash_lo &&
         !execution.steps.front().artifact.source_text.empty() &&
         execution.steps.front().artifact.canonical_ir_bytes.empty() &&
         !execution.steps.front().cpu_input.ok &&
         execution.steps.front().cpu_input.retained_dynamic_memory_bytes() ==
             0u &&
         plan.ok && warm.ok && warm.parse_count == 0u &&
         warm.emission_count == 0u;
}

bool InitializationIsGraphOwned(const Fixture &fixture) {
  std::array<rund::AccelGraphBufferRef, 2u> refs = fixture.refs;
  refs[1u].init = rund::kernel::BufferInit::Zero;
  std::array<rund::AccelGraphNode, 1u> nodes = fixture.nodes;
  nodes[0u].buffers = refs.data();
  const rund::AccelGraph graph{
      .nodes = nodes.data(),
      .node_count = nodes.size(),
      .scalar = fixture.graph.scalar,
      .domain = fixture.graph.domain,
      .fixed_format = fixture.graph.fixed_format,
  };
  const rund::AccelKernel kernel =
      rund::node::accel::CompileAccelKernel(fixture.context, graph);
  if (!kernel.check.ok || (kernel.graph_id_hi == fixture.first.graph_id_hi &&
                           kernel.graph_id_lo == fixture.first.graph_id_lo)) {
    return false;
  }
  const rund::node::accel::detail::KernelExecution execution =
      rund::node::accel::detail::AdmitKernelForExecution(fixture.context,
                                                         kernel);
  return execution.admission.check.ok && execution.resets.size() == 1u &&
         execution.resets.front().binding == 1u &&
         execution.resets.front().step.index == 0u &&
         execution.resets.front().last.index == 0u;
}

bool GraphIdIgnoresBufferIds(const Fixture &fixture) {
  const rund::AccelBuffer padding =
      rund::node::accel::CreateAccelBuffer(fixture.context, BufferDesc());
  const rund::AccelBuffer shifted_input = rund::node::accel::CreateAccelBuffer(
      fixture.context, BufferDesc(rund::BufferUsage::ReadOnly));
  const rund::AccelBuffer shifted_output = rund::node::accel::CreateAccelBuffer(
      fixture.context, BufferDesc(rund::BufferUsage::WriteOnly));
  if (!padding.check.ok || !shifted_input.check.ok ||
      !shifted_output.check.ok ||
      shifted_input.resident.id == fixture.input.resident.id ||
      shifted_output.resident.id == fixture.output.resident.id) {
    return false;
  }

  std::array<rund::AccelGraphBufferRef, 2u> refs{};
  std::array<rund::AccelGraphNode, 1u> nodes{};
  const rund::AccelGraph graph =
      GraphFor(fixture.op.ir(), shifted_input, shifted_output, refs, nodes);
  const rund::AccelKernel shifted =
      rund::node::accel::CompileAccelKernel(fixture.context, graph);
  return shifted.check.ok && shifted.graph_id_hi == fixture.first.graph_id_hi &&
         shifted.graph_id_lo == fixture.first.graph_id_lo;
}

} // namespace node_accel_contract::kernel_case::compile

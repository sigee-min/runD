#pragma once

#include <accel/graph/factory/map.hpp>

#include <accel/context/value.hpp>
#include <accel/graph/value.hpp>
#include <accel/kernel/evidence.hpp>
#include <accel/kernel/run.hpp>
#include <accel/kernel/value.hpp>

#include <node/accel/context.hpp>

#include "resources.hpp"

#include <array>
#include <cstdio>
#include <string_view>

namespace node_accel_contract::fusion::chain {

[[nodiscard]] bool RunFused(const rund::AccelContext &context,
                            const rund::compute_dsl::ComputeOp &op,
                            const Inputs &inputs, const Resources &resources) {
  std::array<GraphBufferRef, 4u> refs{
      GraphBufferRef{.buffer = &resources.fused_input, .role = Role::Read},
      GraphBufferRef{.buffer = &resources.fused_mid,
                     .role = Role::Write,
                     .init = rund::kernel::BufferInit::Zero,
                     .visibility = Visibility::Internal},
      GraphBufferRef{.buffer = &resources.fused_mid,
                     .role = Role::Read,
                     .visibility = Visibility::Internal},
      GraphBufferRef{.buffer = &resources.fused_output,
                     .role = Role::Write,
                     .init = rund::kernel::BufferInit::Zero},
  };
  std::array<GraphNode, 2u> nodes{
      rund::AccelMap(op.ir(), refs.data(), 2u, resources.fused_output.count),
      rund::AccelMap(op.ir(), refs.data() + 2u, 2u,
                     resources.fused_output.count),
  };
  const rund::AccelKernel kernel = rund::node::accel::CompileAccelKernel(
      context, rund::AccelGraph{
                   .nodes = nodes.data(),
                   .node_count = nodes.size(),
                   .scalar = op.ir().scalar,
                   .domain = op.ir().domain,
                   .fixed_format = op.ir().fixed_format,
               });
  if (!kernel.check.ok || kernel.node_count != 2u) {
    std::fprintf(stderr, "fusion compile failed ok=%d reason=%s nodes=%llu\n",
                 kernel.check.ok ? 1 : 0, kernel.check.reason,
                 static_cast<unsigned long long>(kernel.node_count));
    return false;
  }
  const rund::node::accel::detail::KernelExecution execution =
      rund::node::accel::detail::AdmitKernelForExecution(context, kernel);
  if (!execution.admission.check.ok || execution.removed_dispatch_count != 1u ||
      execution.steps.size() != 1u ||
      execution.original_operation_count != 2u ||
      execution.fused_operation_count != 1u ||
      execution.fusion_rejection_count != 0u ||
      std::string_view{execution.fusion_reason} != "compute_fusion_ok" ||
      execution.steps.front().source.begin.index != 0u ||
      execution.steps.front().source.end.index != 2u ||
      execution.resets.size() != 1u || execution.resets.front().binding != 3u ||
      execution.resets.front().step.index != 0u ||
      execution.resets.front().last.index != 0u ||
      !StepArtifactIsChecked(execution.steps.front(), kernel)) {
    std::fprintf(
        stderr,
        "fusion admission failed ok=%d reason=%s removed_dispatch=%llu "
        "steps=%zu "
        "original_ops=%llu fused_ops=%llu rejects=%llu fusion_reason=%s\n",
        execution.admission.check.ok ? 1 : 0, execution.admission.check.reason,
        static_cast<unsigned long long>(execution.removed_dispatch_count),
        execution.steps.size(),
        static_cast<unsigned long long>(execution.original_operation_count),
        static_cast<unsigned long long>(execution.fused_operation_count),
        static_cast<unsigned long long>(execution.fusion_rejection_count),
        execution.fusion_reason);
    return false;
  }
  std::array<KernelBinding, 4u> bindings{
      KernelBinding{.buffer = &resources.fused_input, .role = Role::Read},
      KernelBinding{.buffer = &resources.fused_mid, .role = Role::Write},
      KernelBinding{.buffer = &resources.fused_mid, .role = Role::Read},
      KernelBinding{.buffer = &resources.fused_output, .role = Role::Write},
  };
  const auto evidence_ok = [](const rund::AccelEvidence &evidence) {
    return evidence.ok && evidence.original_operation_count == 2u &&
           evidence.fused_operation_count == 1u &&
           evidence.original_dispatch_count == 2u &&
           evidence.final_dispatch_count == 1u &&
           evidence.dispatch_count == 1u &&
           evidence.internal_producer_consumer_roundtrip_bytes == 0u &&
           evidence.external_producer_consumer_roundtrip_bytes == 0u &&
           evidence.fusion_rejection_count == 0u &&
           std::string_view{evidence.fusion_reason} == "compute_fusion_ok";
  };
  const auto run = [&]() {
    return rund::node::accel::RunAccelKernel(
        context, kernel,
        rund::AccelRun{
            .bindings = bindings.data(),
            .binding_count = bindings.size(),
            .tile_count = inputs.host.size(),
            .fresh_evidence = true,
        });
  };
  const rund::AccelEvidence evidence = run();
  const bool ok = evidence_ok(evidence);
  if (!ok) {
    std::fprintf(
        stderr,
        "fusion evidence failed ok=%d reason=%s original=%llu fused=%llu "
        "original_dispatch=%llu final_dispatch=%llu dispatch=%llu "
        "roundtrip_internal=%llu roundtrip_external=%llu rejects=%llu "
        "fusion_reason=%s\n",
        evidence.ok ? 1 : 0, evidence.reason,
        static_cast<unsigned long long>(evidence.original_operation_count),
        static_cast<unsigned long long>(evidence.fused_operation_count),
        static_cast<unsigned long long>(evidence.original_dispatch_count),
        static_cast<unsigned long long>(evidence.final_dispatch_count),
        static_cast<unsigned long long>(evidence.dispatch_count),
        static_cast<unsigned long long>(
            evidence.internal_producer_consumer_roundtrip_bytes),
        static_cast<unsigned long long>(
            evidence.external_producer_consumer_roundtrip_bytes),
        static_cast<unsigned long long>(evidence.fusion_rejection_count),
        evidence.fusion_reason);
  }
  if (!ok) {
    return false;
  }
  const rund::AccelEvidence warm = run();
  return evidence_ok(warm);
}

} // namespace node_accel_contract::fusion::chain

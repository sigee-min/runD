#include "step.hpp"
#include <kernel/program/compute/lowering/emission.hpp>
#include <kernel/program/compute/plan.hpp>
#include <limits>
#include <utility>

namespace rund::node::accel::detail {
namespace {

[[nodiscard]] KernelExecutionStep BuildKernelExecutionStep(
    rund::kernel::LoweringArtifact artifact,
    rund::kernel::compute_lowering_detail::ComputeInputAdmission input,
    KernelBindingIndices binding_indices, Operation operation,
    const std::uint64_t primitive_hash_hi,
    const std::uint64_t primitive_hash_lo, const std::uint64_t element_count,
    const rund::kernel::GraphControl control) {
  const rund::kernel::NodeKind kind = operation.kind();
  bool binding_indices_ok = true;
  if (kind == rund::kernel::NodeKind::Map) {
    const std::size_t data_binding_count =
        artifact.metadata.binding_accesses.size();
    if (!binding_indices.valid() ||
        data_binding_count > binding_indices.size() ||
        !control.valid(binding_indices.size())) {
      binding_indices_ok = false;
    }
  } else if (!binding_indices.valid()) {
    binding_indices_ok = false;
  }
  if (!binding_indices_ok) {
    artifact.ok = false;
    artifact.reason = "accel_kernel_graph_invalid";
  }
  if (kind == rund::kernel::NodeKind::Map) {
    if (artifact.key.api == rund::kernel::ComputeApi::Cpu) {
      std::string{}.swap(artifact.source_text);
    } else {
      input = {};
    }
  } else {
    input = {};
  }
  return KernelExecutionStep{
      .operation = std::move(operation),
      .artifact = std::move(artifact),
      .cpu_input = std::move(input),
      .graph_binding_indices = std::move(binding_indices),
      .graph_binding_indices_ok = binding_indices_ok,
      .primitive_hash_hi = primitive_hash_hi,
      .primitive_hash_lo = primitive_hash_lo,
      .element_count = element_count,
      .control = control,
  };
}

} // namespace

KernelExecutionStep BuildMapKernelExecutionStep(
    rund::kernel::LoweringArtifact artifact,
    rund::kernel::compute_lowering_detail::ComputeInputAdmission input,
    KernelBindingIndices binding_indices, const std::uint64_t element_count) {
  return BuildKernelExecutionStep(std::move(artifact), std::move(input),
                                  std::move(binding_indices), Operation{}, 0u,
                                  0u, element_count, {});
}

KernelExecutionStep BuildKernelExecutionStep(GraphCompileNode &&node) {
  if (node.kind() == rund::kernel::NodeKind::Map) {
    auto retained = rund::kernel::compute_lowering_detail::
        EmitAdmittedRetainedComputeArtifact(std::move(node.map_metadata),
                                            std::move(node.cpu_input));
    return BuildKernelExecutionStep(
        std::move(retained.artifact), std::move(retained.input),
        std::move(node.binding_indices), std::move(node.operation),
        node.primitive_hash_hi, node.primitive_hash_lo, node.element_count,
        node.control);
  }
  return BuildKernelExecutionStep(
      std::move(node.artifact), std::move(node.cpu_input),
      std::move(node.binding_indices), std::move(node.operation),
      node.primitive_hash_hi, node.primitive_hash_lo, node.element_count,
      node.control);
}

FrozenDispatchCount
BuildMapDispatchCount(const rund::kernel::ExecutionMetadata &metadata,
                      const std::uint64_t element_count,
                      const rund::kernel::ComputeCaps &caps,
                      const std::uint64_t phase_id) {
  if (!metadata.ok || element_count == 0u || phase_id == 0u) {
    return FrozenDispatchCount{.reason = "accel_kernel_graph_invalid"};
  }
  const rund::kernel::ComputeLimit limit{
      .staging_bytes = caps.staging_bytes,
      .max_window_tiles = caps.max_window_tiles,
  };
  rund::kernel::ComputeMap map = metadata.map;
  map.api = caps.api;
  const rund::kernel::TilePhaseDescription phase{
      .phase_id = phase_id,
      .tile_count = element_count,
      .capacity =
          rund::kernel::TilePhaseCapacityRequirement{
              .scratch_alignment = 1u,
              .output_shards = element_count,
              .queue_slots = element_count,
              .task_slots = element_count,
          },
  };
  const rund::kernel::ComputeDispatchPlan dispatch =
      rund::kernel::PlanComputeDispatch(phase, map, caps, limit);
  if (!dispatch.ok || dispatch.dispatch_count == 0u) {
    return FrozenDispatchCount{.reason = dispatch.reason};
  }
  return FrozenDispatchCount{
      .count = dispatch.dispatch_count,
      .ok = true,
      .reason = "ok",
  };
}

FrozenDispatchCount
BuildOriginalDispatchCount(const std::span<const GraphCompileNode> nodes,
                           const rund::kernel::ComputeCaps &caps,
                           const std::uint64_t phase_offset) {
  FrozenDispatchCount result{};
  for (std::size_t index = 0u; index < nodes.size(); ++index) {
    const GraphCompileNode &node = nodes[index];
    if (node.kind() != rund::kernel::NodeKind::Map) {
      result.reason = "accel_kernel_graph_invalid";
      return result;
    }
    const FrozenDispatchCount dispatch = BuildMapDispatchCount(
        node.map_metadata, node.element_count, caps,
        phase_offset + static_cast<std::uint64_t>(index) + 1u);
    if (!dispatch.ok) {
      result.reason = dispatch.reason;
      return result;
    }
    if (result.count >
        std::numeric_limits<std::uint64_t>::max() - dispatch.count) {
      result.reason = "compute_dispatch_overflow";
      return result;
    }
    result.count += dispatch.count;
  }
  result.ok = result.count != 0u;
  result.reason = result.ok ? "ok" : "compute_plan_invalid";
  return result;
}

} // namespace rund::node::accel::detail

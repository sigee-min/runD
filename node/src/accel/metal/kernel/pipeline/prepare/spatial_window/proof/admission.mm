#include "internal.hpp"

#include "../../../build.hpp"
#include "../../../state.hpp"

#include "../../../../../../kernel/recurrence/plan.hpp"
#include "../../../../../../kernel/recurrence/source.hpp"

#include <stdexcept>

namespace rund::node::accel::detail {

#if defined(__APPLE__) && defined(RUND_NODE_HAVE_METAL_SDK)

[[nodiscard]] MetalSpatialWindowProof
ProveMetalSpatialWindow(const std::span<const BackendBatchEntry> entries,
                        const MapRecurrence &recurrence,
                        const std::span<const TileTransducer> transducers,
                        const std::span<const NestedAggregate> aggregates,
                        const std::span<const BackendPublish> publications,
                        const std::uint32_t local_count) {
  MetalSpatialWindowProof proof{.local_count = local_count,
                                .step_count = 3u,
                                .producer_index = 0u,
                                .window = 1u,
                                .consumer_index = 2u};
  const BackendRun *const first_run =
      entries.empty() ? nullptr : entries.front().run;
  const KernelExecution *const execution =
      first_run == nullptr ? nullptr : first_run->execution;
  if (execution != nullptr) {
    for (const KernelExecutionStep &step : execution->steps) {
      proof.window_present =
          proof.window_present || step.kind() == rund::kernel::NodeKind::Window;
    }
  }
  if (entries.empty()) {
    metal_spatial_window_proof_internal::RecordDiagnostic(
        metal_spatial_window_proof_internal::DiagnosticReason::EntryNull, 0u,
        local_count, 0u, 0u);
    return proof;
  }
  if (local_count == 0u || local_count > PreparedPipelineStepCapacity) {
    metal_spatial_window_proof_internal::RecordDiagnostic(
        metal_spatial_window_proof_internal::DiagnosticReason::ExecutionShape,
        local_count, PreparedPipelineStepCapacity, 0u, 0u);
    return proof;
  }
  if (recurrence.ready() || recurrence.invalid() ||
      recurrence.history != nullptr || !transducers.empty() ||
      !aggregates.empty() || !publications.empty()) {
    metal_spatial_window_proof_internal::RecordDiagnostic(
        metal_spatial_window_proof_internal::DiagnosticReason::Recurrence,
        recurrence.ready() ? 1u : 0u, recurrence.invalid() ? 1u : 0u,
        recurrence.history != nullptr ? 1u : 0u,
        static_cast<std::uint64_t>(transducers.size() + aggregates.size() +
                                   publications.size()));
    return proof;
  }

  if (execution == nullptr || execution->steps.size() != 3u ||
      execution->resets.size() != 0u ||
      execution->steps[0u].kind() != rund::kernel::NodeKind::Map ||
      execution->steps[1u].kind() != rund::kernel::NodeKind::Window ||
      execution->steps[2u].kind() != rund::kernel::NodeKind::Map) {
    metal_spatial_window_proof_internal::RecordDiagnostic(
        metal_spatial_window_proof_internal::DiagnosticReason::ExecutionShape,
        execution == nullptr
            ? 0u
            : static_cast<std::uint64_t>(execution->steps.size()),
        execution == nullptr
            ? 0u
            : static_cast<std::uint64_t>(execution->resets.size()),
        execution == nullptr || execution->steps.size() < 2u
            ? 0u
            : static_cast<std::uint64_t>(execution->steps[1u].kind()),
        0u);
    return proof;
  }

  bool shape_set = false;
  bool graph_binding_set = false;
  bool binding_set = false;
  for (const BackendBatchEntry &entry : entries) {
    if (!metal_spatial_window_proof_internal::CaptureGeometry(
            entry, *execution, proof, shape_set, graph_binding_set) ||
        !metal_spatial_window_proof_internal::CaptureBindings(entry, proof,
                                                              binding_set)) {
      return proof;
    }
  }
  proof.admitted = shape_set && binding_set && proof.graph_bindings_valid();
  if (proof.admitted) {
    proof.locals.resize(local_count);
    if (proof.locals.capacity() != local_count) {
      throw std::length_error("spatial Window proof capacity");
    }
  }
  return proof;
}

#endif

} // namespace rund::node::accel::detail

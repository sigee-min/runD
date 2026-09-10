#include "internal.hpp"

#include "../../../build.hpp"
#include "../../../state.hpp"

#include "../../../../../../kernel/backend/exception.hpp"

#include "../../../../../runtime/map/api.hpp"
#include "../../../../../runtime/map/resources.hpp"

#include "../../../../../../kernel/recurrence/plan.hpp"
#include "../../../../../../kernel/recurrence/source.hpp"

namespace rund::node::accel::detail::metal_spatial_window_proof_internal {

#if defined(__APPLE__) && defined(RUND_NODE_HAVE_METAL_SDK)

[[nodiscard]] bool CaptureGeometry(const BackendBatchEntry &entry,
                                   const KernelExecution &execution,
                                   MetalSpatialWindowProof &proof,
                                   bool &shape_set,
                                   bool &graph_binding_set) noexcept {
  const BackendRun *const run = entry.run;
  if (run == nullptr || run->execution == nullptr || run->steps == nullptr) {
    RecordDiagnostic(DiagnosticReason::EntryNull, entry.occurrence_index,
                     run == nullptr ? 0u : 1u,
                     run == nullptr || run->execution == nullptr ? 0u : 1u,
                     run == nullptr || run->steps == nullptr ? 0u : 1u);
    return false;
  }
  if (!MetalSpatialWindowSameKernelExecution(execution, *run->execution)) {
    RecordDiagnostic(
        DiagnosticReason::EntrySemanticMismatch, entry.occurrence_index,
        static_cast<std::uint64_t>(execution.steps.size()),
        static_cast<std::uint64_t>(run->execution->steps.size()), 0u);
    return false;
  }
  if (run->step_count != run->execution->steps.size()) {
    RecordDiagnostic(DiagnosticReason::DeclaredRange, entry.occurrence_index,
                     run->step_count,
                     static_cast<std::uint64_t>(run->execution->steps.size()),
                     0u);
    return false;
  }
  if (run->resets != nullptr && !run->resets->empty()) {
    RecordDiagnostic(DiagnosticReason::Recurrence, entry.occurrence_index,
                     run->resets->size(), 0u, 0u);
    return false;
  }
  if (entry.recurrence.window != nullptr ||
      entry.recurrence.writes_each_iteration ||
      entry.transducer != NoTileTransducer) {
    RecordDiagnostic(DiagnosticReason::Recurrence, entry.occurrence_index,
                     entry.recurrence.window != nullptr ? 1u : 0u,
                     entry.recurrence.writes_each_iteration ? 1u : 0u,
                     entry.transducer != NoTileTransducer ? 1u : 0u);
    return false;
  }

  for (std::size_t index = 0u; index < run->step_count; ++index) {
    const BoundStep &bound = run->steps[index];
    const KernelExecutionStep *const step = &run->execution->steps[index];
    if (bound.index != index || bound.step != step ||
        !MetalSpatialWindowPlannedStepValid(bound, *step) ||
        bound.source_binds == nullptr || bound.control.active() ||
        !bound.resets.empty() || !BoundStepMatches(bound, step->kind()) ||
        !step->graph_binding_indices_ok ||
        !step->graph_binding_indices.valid() ||
        step->graph_binding_indices.size() != 2u) {
      RecordDiagnostic(DiagnosticReason::DeclaredRange, entry.occurrence_index,
                       index, bound.index, step->graph_binding_indices.size());
      return false;
    }
    for (std::size_t binding = 0u; binding < 2u; ++binding) {
      const std::size_t proof_index = index * 2u + binding;
      if (!graph_binding_set) {
        proof.binding_indices[proof_index] =
            step->graph_binding_indices[binding];
      } else if (proof.binding_indices[proof_index] !=
                 step->graph_binding_indices[binding]) {
        RecordDiagnostic(DiagnosticReason::DeclaredRange,
                         entry.occurrence_index, proof_index,
                         proof.binding_indices[proof_index],
                         step->graph_binding_indices[binding]);
        return false;
      }
    }
    if (index == 0u || index == 2u) {
      if (OperationFor<operation::Map>(bound) == nullptr) {
        RecordDiagnostic(DiagnosticReason::TopologyOperator,
                         entry.occurrence_index, index, 0u, 0u);
        return false;
      }
      const StepBinds *const bindings = BindingsFor<StepBinds>(bound);
      if (bindings == nullptr || !bindings->valid() ||
          bindings->inputs.size() != 1u || bindings->outputs.size() != 1u) {
        RecordDiagnostic(DiagnosticReason::TopologyOperator,
                         entry.occurrence_index, index,
                         bindings == nullptr ? 0u : bindings->inputs.size(),
                         bindings == nullptr ? 0u : bindings->outputs.size());
        return false;
      }
      continue;
    }

    const operation::Window *const window =
        OperationFor<operation::Window>(bound);
    const RangeBinds *const bindings =
        BindingsFor<RangeBinds>(bound, rund::kernel::NodeKind::Window);
    if (window == nullptr || bindings == nullptr) {
      RecordDiagnostic(
          DiagnosticReason::TopologyOperator, entry.occurrence_index, index,
          window == nullptr ? 0u : 1u, bindings == nullptr ? 0u : 1u);
      return false;
    }
    if (window->plan.count_source !=
            rund::kernel::ComputeCountSource::Descriptor ||
        !window->range.ok()) {
      RecordDiagnostic(DiagnosticReason::TopologyOperator,
                       entry.occurrence_index, index,
                       static_cast<std::uint64_t>(window->plan.count_source),
                       window->range.ok() ? 1u : 0u);
      return false;
    }
    if (window->range.candidate().disposition() != RangePath::SharedHalo ||
        window->range.stage_count() != 1u ||
        !window->range.shape().centered_clamp() ||
        window->range.shape().stride() != 1u ||
        (window->range.shape().element_bytes() != 4u &&
         window->range.shape().element_bytes() != 8u)) {
      RecordDiagnostic(
          DiagnosticReason::SharedHaloCandidate, entry.occurrence_index,
          static_cast<std::uint64_t>(window->range.candidate().disposition()),
          static_cast<std::uint64_t>(window->range.stage_count()),
          window->range.shape().stride());
      return false;
    }
    if (bindings->input == nullptr || bindings->output == nullptr ||
        bindings->input->id == 0u || bindings->output->id == 0u ||
        bindings->input->bytes == 0u || bindings->output->bytes == 0u ||
        bindings->input->stride_bytes < bindings->input->element_bytes ||
        bindings->output->stride_bytes < bindings->output->element_bytes) {
      RecordDiagnostic(DiagnosticReason::BindingCapture, entry.occurrence_index,
                       bindings->input == nullptr ? 0u : 1u,
                       bindings->output == nullptr ? 0u : 1u,
                       bindings->input == nullptr || bindings->output == nullptr
                           ? 0u
                           : bindings->input->bytes + bindings->output->bytes);
      return false;
    }
    if (bindings->input_handle == nullptr ||
        bindings->output_handle == nullptr) {
      RecordDiagnostic(DiagnosticReason::BindingHandle, entry.occurrence_index,
                       bindings->input_handle == nullptr ? 0u : 1u,
                       bindings->output_handle == nullptr ? 0u : 1u, 0u);
      return false;
    }
    if (*bindings->input_handle == nullptr ||
        *bindings->output_handle == nullptr) {
      RecordDiagnostic(DiagnosticReason::BindingHandle, entry.occurrence_index,
                       *bindings->input_handle == nullptr ? 0u : 1u,
                       *bindings->output_handle == nullptr ? 0u : 1u, 0u);
      return false;
    }

    const RangeShape &shape = window->range.shape();
    const RangeCandidate &candidate = window->range.candidate();
    if (!shape_set) {
      proof.execution = &execution;
      proof.descriptor = MetalSpatialWindowDescriptorKey(*window);
      proof.plan = MetalSpatialWindowPlanKey(*window);
      proof.source_identity = window->range.source_identity();
      proof.execution_identity = window->range.execution_identity();
      proof.candidate_identity = window->range.source_identity();
      proof.candidate_path = candidate.disposition();
      proof.candidate_disposition =
          static_cast<std::uint8_t>(candidate.disposition());
      proof.range = MetalSpatialWindowRangeKey(window->range);
      proof.window_input_binding = step->graph_binding_indices[0u];
      proof.window_output_binding = step->graph_binding_indices[1u];
      proof.candidate_width = candidate.width();
      proof.candidate_radius = candidate.radius_capacity();
      proof.halo_radius = candidate.radius_capacity();
      proof.halo_frame = MetalSpatialWindowHaloFrame(candidate);
      proof.halo_payload = MetalSpatialWindowHaloPayload(candidate, shape);
      proof.halo_boundary = static_cast<std::uint8_t>(shape.boundary());
      proof.candidate_captured = true;
      proof.halo_captured = true;
      proof.stage_count =
          static_cast<std::uint32_t>(window->range.stage_count());
      proof.temporary_count =
          static_cast<std::uint32_t>(window->range.temporary_count());
      proof.source_variant =
          static_cast<std::uint8_t>(window->range.source_variant());
      shape_set = true;
    } else if (proof.descriptor != MetalSpatialWindowDescriptorKey(*window) ||
               proof.plan != MetalSpatialWindowPlanKey(*window) ||
               proof.source_identity != window->range.source_identity() ||
               proof.execution_identity != window->range.execution_identity() ||
               proof.candidate_identity != window->range.source_identity() ||
               proof.candidate_path != candidate.disposition() ||
               proof.candidate_disposition !=
                   static_cast<std::uint8_t>(candidate.disposition()) ||
               proof.candidate_width != candidate.width() ||
               proof.candidate_radius != candidate.radius_capacity() ||
               proof.halo_radius != candidate.radius_capacity() ||
               proof.halo_frame != MetalSpatialWindowHaloFrame(candidate) ||
               proof.halo_payload !=
                   MetalSpatialWindowHaloPayload(candidate, shape) ||
               proof.halo_boundary !=
                   static_cast<std::uint8_t>(shape.boundary()) ||
               proof.stage_count != window->range.stage_count() ||
               proof.temporary_count != window->range.temporary_count() ||
               proof.source_variant !=
                   static_cast<std::uint8_t>(window->range.source_variant()) ||
               proof.range != MetalSpatialWindowRangeKey(window->range) ||
               shape.input_count() != proof.plan[6u] ||
               shape.output_count() != proof.plan[7u] ||
               shape.window_size() != proof.plan[8u] ||
               shape.stride() != proof.plan[9u] ||
               shape.padding() != proof.plan[10u] ||
               shape.element_bytes() != proof.plan[11u]) {
      RecordDiagnostic(DiagnosticReason::SharedHaloIdentity,
                       entry.occurrence_index, proof.candidate_width,
                       candidate.width(), proof.candidate_radius);
      return false;
    }
  }
  graph_binding_set = true;
  return true;
}

#endif

} // namespace rund::node::accel::detail::metal_spatial_window_proof_internal

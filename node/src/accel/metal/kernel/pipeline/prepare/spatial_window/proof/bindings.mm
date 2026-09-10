#include "internal.hpp"

#include "../../../build.hpp"
#include "../../../state.hpp"

#include "../../../../../../kernel/backend/exception.hpp"

#include "../../../../../runtime/map/api.hpp"
#include "../../../../../runtime/map/resources.hpp"

#include "../../../../../../kernel/recurrence/plan.hpp"
#include "../../../../../../kernel/recurrence/source.hpp"

#include <array>

namespace rund::node::accel::detail::metal_spatial_window_proof_internal {

#if defined(__APPLE__) && defined(RUND_NODE_HAVE_METAL_SDK)

[[nodiscard]] bool CaptureBindings(const BackendBatchEntry &entry,
                                   MetalSpatialWindowProof &proof,
                                   bool &binding_set) noexcept {
  const BackendRun *const run = entry.run;
  const BoundStep &producer_step = run->steps[0u];
  const BoundStep &window_step = run->steps[1u];
  const BoundStep &consumer_step = run->steps[2u];
  const StepBinds *const producer_binds =
      BindingsFor<StepBinds>(producer_step, rund::kernel::NodeKind::Map);
  const RangeBinds *const window_binds =
      BindingsFor<RangeBinds>(window_step, rund::kernel::NodeKind::Window);
  const StepBinds *const consumer_binds =
      BindingsFor<StepBinds>(consumer_step, rund::kernel::NodeKind::Map);
  if (producer_binds == nullptr || window_binds == nullptr ||
      consumer_binds == nullptr) {
    RecordDiagnostic(DiagnosticReason::BindingCapture, entry.occurrence_index,
                     producer_binds == nullptr ? 0u : 1u,
                     window_binds == nullptr ? 0u : 1u,
                     consumer_binds == nullptr ? 0u : 1u);
    return false;
  }
  if (!producer_binds->valid() || !consumer_binds->valid() ||
      producer_binds->inputs.size() != 1u ||
      producer_binds->outputs.size() != 1u ||
      consumer_binds->inputs.size() != 1u ||
      consumer_binds->outputs.size() != 1u) {
    RecordDiagnostic(DiagnosticReason::BindingCapture, entry.occurrence_index,
                     producer_binds->inputs.size(),
                     producer_binds->outputs.size(),
                     consumer_binds->inputs.size());
    return false;
  }
  if (producer_step.source_binds == nullptr ||
      producer_step.source_binds != window_step.source_binds ||
      consumer_step.source_binds != window_step.source_binds) {
    RecordDiagnostic(DiagnosticReason::BindingOwner, entry.occurrence_index,
                     producer_step.source_binds == nullptr ? 0u : 1u,
                     window_step.source_binds == nullptr ? 0u : 1u,
                     consumer_step.source_binds == nullptr ? 0u : 1u);
    return false;
  }
  if (window_binds->input == nullptr || window_binds->output == nullptr) {
    RecordDiagnostic(DiagnosticReason::BindingCapture, entry.occurrence_index,
                     window_binds->input == nullptr ? 0u : 1u,
                     window_binds->output == nullptr ? 0u : 1u, 0u);
    return false;
  }
  if (window_binds->input_handle == nullptr ||
      window_binds->output_handle == nullptr) {
    RecordDiagnostic(DiagnosticReason::BindingHandle, entry.occurrence_index,
                     window_binds->input_handle == nullptr ? 0u : 1u,
                     window_binds->output_handle == nullptr ? 0u : 1u, 0u);
    return false;
  }
  if (*window_binds->input_handle == nullptr ||
      *window_binds->output_handle == nullptr) {
    RecordDiagnostic(DiagnosticReason::BindingHandle, entry.occurrence_index,
                     *window_binds->input_handle == nullptr ? 0u : 1u,
                     *window_binds->output_handle == nullptr ? 0u : 1u, 0u);
    return false;
  }
  const rund::kernel::ResidentBindingRange producer_inputs =
      producer_binds->inputs.range();
  const rund::kernel::ResidentBindingRange producer_outputs =
      producer_binds->outputs.range();
  const rund::kernel::ResidentBindingRange consumer_inputs =
      consumer_binds->inputs.range();
  const rund::kernel::ResidentBindingRange consumer_outputs =
      consumer_binds->outputs.range();
  const std::array<const rund::kernel::ResidentBufferRef *, 6u> refs{
      producer_inputs.ref(0u), producer_outputs.ref(0u),
      window_binds->input,     window_binds->output,
      consumer_inputs.ref(0u), consumer_outputs.ref(0u)};
  const std::array<const std::shared_ptr<void> *, 6u> handles{
      producer_inputs.handle(0u), producer_outputs.handle(0u),
      window_binds->input_handle, window_binds->output_handle,
      consumer_inputs.handle(0u), consumer_outputs.handle(0u)};
  if (!proof.graph_bindings_valid()) {
    RecordDiagnostic(DiagnosticReason::BindingAlias, entry.occurrence_index,
                     proof.binding_indices[0u], proof.binding_indices[1u],
                     proof.binding_indices[2u]);
    return false;
  }
  constexpr std::array<std::uint32_t, 6u> expected_usage{
      rund::kernel::kResidentUsageRead, rund::kernel::kResidentUsageWrite,
      rund::kernel::kResidentUsageRead, rund::kernel::kResidentUsageWrite,
      rund::kernel::kResidentUsageRead, rund::kernel::kResidentUsageWrite};
  for (std::size_t edge = 0u; edge < refs.size(); ++edge) {
    if (refs[edge] == nullptr || handles[edge] == nullptr ||
        *handles[edge] == nullptr ||
        !MetalSpatialWindowResidentRefValid(*refs[edge]) ||
        refs[edge]->usage != expected_usage[edge]) {
      RecordDiagnostic(
          DiagnosticReason::BindingCapture, entry.occurrence_index, edge,
          refs[edge] == nullptr ? 0u : refs[edge]->usage,
          handles[edge] == nullptr || *handles[edge] == nullptr ? 0u : 1u);
      return false;
    }
  }
  if (!MetalSpatialWindowResidentRefPhysicalEqual(*refs[1u], *refs[2u]) ||
      !MetalSpatialWindowResidentRefPhysicalEqual(*refs[3u], *refs[4u]) ||
      handles[1u]->get() != handles[2u]->get() ||
      handles[3u]->get() != handles[4u]->get()) {
    RecordDiagnostic(
        DiagnosticReason::BindingAlias, entry.occurrence_index,
        MetalSpatialWindowResidentRefPhysicalEqual(*refs[1u], *refs[2u]) ? 1u
                                                                         : 0u,
        MetalSpatialWindowResidentRefPhysicalEqual(*refs[3u], *refs[4u]) ? 1u
                                                                         : 0u,
        handles[1u]->get() == handles[2u]->get() &&
                handles[3u]->get() == handles[4u]->get()
            ? 1u
            : 0u);
    return false;
  }
  // Keep binding_set closed until producer, Window, and consumer have all
  // supplied authenticated references and handles for this entry.
  binding_set = true;
  return true;
}

#endif

} // namespace rund::node::accel::detail::metal_spatial_window_proof_internal

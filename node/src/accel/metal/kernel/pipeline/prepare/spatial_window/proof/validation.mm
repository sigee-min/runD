#include "../../../build.hpp"
#include "../../../state.hpp"

namespace rund::node::accel::detail {

#if defined(__APPLE__) && defined(RUND_NODE_HAVE_METAL_SDK)

[[nodiscard]] bool MetalSequence::spatial_window_proof_valid() const noexcept {
  if (!persistent_spatial_window_selectable || !spatial_window.admitted ||
      spatial_window.execution == nullptr || spatial_window.local_count == 0u ||
      residency_steps.size() != spatial_window.local_count ||
      state_count != 0u || states != nil ||
      residency_prefix.dispatch_count != 0u ||
      residency_suffix.dispatch_count != 0u || !spatial_window.shape_valid() ||
      !spatial_window.operation_valid()) {
    return false;
  }
  for (std::size_t index = 0u; index < spatial_window.step_count; ++index) {
    const KernelExecutionStep &step = spatial_window.execution->steps[index];
    if (!step.graph_binding_indices_ok || !step.graph_binding_indices.valid() ||
        step.graph_binding_indices.size() != 2u ||
        step.graph_binding_indices[0u] !=
            spatial_window.binding_indices[index * 2u] ||
        step.graph_binding_indices[1u] !=
            spatial_window.binding_indices[index * 2u + 1u]) {
      return false;
    }
  }
  for (std::size_t local = 0u; local < residency_steps.size(); ++local) {
    const MetalResidencyStepRange &range = residency_steps[local];
    if (!range.valid() || range.dispatch_count == 0u ||
        spatial_window.locals[local].range_begin != range.begin ||
        spatial_window.locals[local].range_count != range.count ||
        spatial_window.locals[local].range_dispatch_count !=
            range.dispatch_count) {
      return false;
    }
    if (!spatial_window.local_matches(local)) {
      return false;
    }
  }
  return true;
}

#endif

} // namespace rund::node::accel::detail

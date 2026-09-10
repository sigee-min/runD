#include "internal.hpp"

#include "../../../recurrence/match.hpp"

namespace rund::node::accel::detail::device_vsm_projection {

bool exact_pointwise_pipeline(const prepared::PipelineState &pipeline,
                              const BoundStep *&first,
                              rund::kernel::BindingSet &bindings,
                              const char *&reason) noexcept {
  reason = "device_vsm_primary_pipeline_invalid";
  if (pipeline.state_count == 0u || pipeline.states == nullptr ||
      pipeline.size == 0u) {
    reason = "device_vsm_pipeline_owner_invalid";
    return false;
  }
  first = device_vsm_window_projection::map_step(pipeline.states[0u].get(),
                                                 false);
  if (first == nullptr) {
    reason = "device_vsm_pipeline_map_invalid";
    return false;
  }
  bindings = MapBindingFor(*first);
  if (!bindings.ok) {
    reason = "device_vsm_pipeline_binding_unavailable";
    return false;
  }
  const rund::kernel::ComputePlan &plan = first->planned->plan;
  if (plan.input_buffer_count == 0u ||
      plan.input_buffer_count >= DeviceVsmResidentCapacity ||
      plan.output_buffer_count != 1u ||
      bindings.resident_inputs.count != plan.input_buffer_count ||
      bindings.resident_outputs.count != plan.output_buffer_count ||
      first->step->artifact.metadata.read_count != plan.input_buffer_count ||
      first->step->artifact.metadata.write_count != plan.output_buffer_count) {
    reason = "device_vsm_pipeline_resident_count_invalid";
    return false;
  }
  if (bindings.param_data_bytes != bindings.param_bytes ||
      (bindings.param_bytes != 0u && bindings.param_data == nullptr) ||
      first->planned->plan.param_bytes != bindings.param_bytes) {
    reason = "device_vsm_pipeline_parameter_invalid";
    return false;
  }
  if (first->planned->plan.dispatch_count != first->map_windows.size()) {
    reason = "device_vsm_pipeline_dispatch_count_invalid";
    return false;
  }
  for (std::size_t index = 1u; index < pipeline.state_count; ++index) {
    const BoundStep *const step = device_vsm_window_projection::map_step(
        pipeline.states[index].get(), false);
    if (step == nullptr ||
        !SamePlan(first->planned->plan, step->planned->plan) ||
        !SameArtifact(first->step->artifact, step->step->artifact) ||
        !SameWindows(*first, *step)) {
      return false;
    }
    const rund::kernel::BindingSet candidate = MapBindingFor(*step);
    if (!candidate.ok || !SameMapBinding(bindings, candidate) ||
        !device_vsm_window_projection::same_parameters(bindings, candidate)) {
      reason = "device_vsm_pipeline_state_mismatch";
      return false;
    }
  }
  reason = "ok";
  return true;
}

bool exact_graph_map_pipeline(const prepared::PipelineState &pipeline,
                              const BoundStep *&first,
                              rund::kernel::BindingSet &bindings,
                              const char *&reason) noexcept {
  reason = "device_vsm_graph_pipeline_invalid";
  if (pipeline.state_count == 0u || pipeline.states == nullptr ||
      pipeline.size == 0u) {
    reason = "device_vsm_graph_pipeline_owner_invalid";
    return false;
  }
  first = device_vsm_window_projection::map_step(pipeline.states[0u].get(),
                                                 true);
  if (first == nullptr) {
    reason = "device_vsm_graph_map_invalid";
    return false;
  }
  if (first->step->artifact.metadata.read_count == 0u ||
      first->step->artifact.metadata.write_count != 1u) {
    reason = "device_vsm_graph_map_binding_invalid";
    return false;
  }
  if (first->planned->plan.input_buffer_count == 0u ||
      first->planned->plan.input_buffer_count >= DeviceVsmResidentCapacity ||
      first->planned->plan.output_buffer_count != 1u ||
      first->step->artifact.metadata.read_count !=
          first->planned->plan.input_buffer_count) {
    reason = "device_vsm_graph_map_plan_invalid";
    return false;
  }
  bindings = MapBindingFor(*first);
  if (!bindings.ok ||
      bindings.resident_inputs.count !=
          first->planned->plan.input_buffer_count ||
      bindings.resident_outputs.count != 1u ||
      bindings.param_bytes != bindings.param_data_bytes ||
      (bindings.param_bytes != 0u && bindings.param_data == nullptr) ||
      bindings.param_bytes != first->planned->plan.param_bytes) {
    reason = "device_vsm_graph_map_parameter_invalid";
    return false;
  }
  for (std::size_t index = 1u; index < pipeline.state_count; ++index) {
    const BoundStep *const step = device_vsm_window_projection::map_step(
        pipeline.states[index].get(), true);
    if (step == nullptr ||
        !SamePlan(first->planned->plan, step->planned->plan) ||
        !SameArtifact(first->step->artifact, step->step->artifact) ||
        !SameWindows(*first, *step)) {
      return false;
    }
    const rund::kernel::BindingSet candidate = MapBindingFor(*step);
    if (!candidate.ok || !SameMapBinding(bindings, candidate) ||
        !device_vsm_window_projection::same_parameters(bindings, candidate)) {
      reason = "device_vsm_graph_map_parameter_mismatch";
      return false;
    }
  }
  reason = "ok";
  return true;
}

bool same_graph_map_pipeline(const prepared::PipelineState &pipeline,
                             const BoundStep &first,
                             const rund::kernel::BindingSet &bindings) noexcept {
  const BoundStep *candidate = nullptr;
  rund::kernel::BindingSet candidate_bindings{};
  const char *reason = nullptr;
  return exact_graph_map_pipeline(pipeline, candidate, candidate_bindings,
                                  reason) &&
         candidate != nullptr &&
         SamePlan(first.planned->plan, candidate->planned->plan) &&
         SameArtifact(first.step->artifact, candidate->step->artifact) &&
         SameWindows(first, *candidate) &&
         device_vsm_window_projection::same_parameters(bindings,
                                                        candidate_bindings);
}

bool same_pointwise_pipeline(const prepared::PipelineState &pipeline,
                             const BoundStep &first,
                             const rund::kernel::BindingSet &bindings) noexcept {
  const BoundStep *peer = nullptr;
  rund::kernel::BindingSet peer_bindings{};
  const char *reason = nullptr;
  return exact_pointwise_pipeline(pipeline, peer, peer_bindings, reason) &&
         peer != nullptr &&
         SamePlan(first.planned->plan, peer->planned->plan) &&
         SameArtifact(first.step->artifact, peer->step->artifact) &&
         SameWindows(first, *peer) && SameMapBinding(bindings, peer_bindings) &&
         device_vsm_window_projection::same_parameters(bindings,
                                                        peer_bindings);
}

} // namespace rund::node::accel::detail::device_vsm_projection

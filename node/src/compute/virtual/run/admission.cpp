#include "admission.hpp"

#include "../backing.hpp"

#include "../../backend.hpp"

namespace rund::compute::detail {

namespace {

[[nodiscard]] Status
virtual_device_capability(const VirtualPipelineState &state) noexcept {
  if (state.pipeline == nullptr || state.pipeline->device == nullptr ||
      state.pipeline->device->backend == Backend::Cpu) {
    return Status::success();
  }
  const DeviceOps *const ops = state.pipeline->device->ops;
  return ops == nullptr ||
                 ops->virtual_execution.virtual_pipeline_capability == nullptr
             ? Status::fail(Reason::BackendUnsupported)
             : ops->virtual_execution.virtual_pipeline_capability(
                   *state.pipeline->device);
}

} // namespace

VirtualRunAdmission admit_virtual_run(const VirtualPipelineState &state,
                                      const VirtualRunProjection &run,
                                      const VirtualBacking &input,
                                      const VirtualBacking &output) noexcept {
  const bool recurrent =
      run.clamp_window && !run.clip_window && !run.device_vsm_required &&
      !run.poolless_device_vsm() && run.input_count == 1u &&
      run.active.stream.epoch_count() >= 2u &&
      input.max_parallel_reads() <= 1u &&
      VirtualBackingAccess::resident(input) == nullptr &&
      VirtualBackingAccess::resident(output) == nullptr &&
      run.input_backing != 0u && run.output_backing != 0u &&
      run.input_backing == VirtualBackingAccess::id(input) &&
      run.output_backing == VirtualBackingAccess::id(output) &&
      run.input_version == VirtualBackingAccess::version(input) &&
      run.output_version == VirtualBackingAccess::version(output);

  const bool transaction_scan =
      run.scan() && !run.graph_execution() && !run.reduction() &&
      !run.device_vsm_required && !run.poolless_device_vsm() &&
      run.input_count == 1u && state.pipeline != nullptr &&
      state.pipeline->device != nullptr &&
      state.pipeline->device->backend != Backend::Cpu &&
      run.active.output_bytes != 0u &&
      run.active.output_bytes == output.size_bytes() &&
      run.host_output_frame_capacity != 0u &&
      VirtualBackingAccess::resident(input) == nullptr &&
      VirtualBackingAccess::resident(output) == nullptr &&
      run.input_backing != 0u && run.output_backing != 0u &&
      run.input_backing == VirtualBackingAccess::id(input) &&
      run.output_backing == VirtualBackingAccess::id(output) &&
      run.input_version == VirtualBackingAccess::version(input) &&
      run.output_version == VirtualBackingAccess::version(output) &&
      dynamic_cast<const VirtualBackingTransaction *>(&output) != nullptr;

  return VirtualRunAdmission{
      .transaction_window = false,
      .transaction_scan = transaction_scan,
      .try_recurrent =
          state.geometry.route != VirtualRoute::Window || recurrent,
  };
}

VirtualRunAdmission
admit_virtual_run(const std::shared_ptr<VirtualPipelineState> &state,
                  const std::uint64_t active_count,
                  std::unique_lock<std::mutex> &state_lock,
                  const VirtualRunAdmissionMode mode) noexcept {
  VirtualRunAdmission result{};
  if (state == nullptr || !valid_virtual_pipeline(state)) {
    return result;
  }
  if (mode == VirtualRunAdmissionMode::AsynchronousDeviceVsm &&
      (state->pipeline == nullptr || state->pipeline->device == nullptr ||
       state->pipeline->device->backend == Backend::Cpu ||
       !state->geometry.device_vsm_required)) {
    result.status = Status::fail(Reason::BackendUnsupported);
    return result;
  }
  if (!state_lock.owns_lock()) {
    state_lock = std::unique_lock<std::mutex>{state->gate, std::try_to_lock};
  }
  if (!state_lock.owns_lock() ||
      state->phase == VirtualPipelinePhase::Running) {
    result.status = Status::fail(Reason::PipelineBusy);
    return result;
  }
  if (state->phase == VirtualPipelinePhase::Poisoned) {
    const Status capability = virtual_device_capability(*state);
    result.status =
        capability ? Status::fail(Reason::PipelinePoisoned) : capability;
    result.poison_pipeline =
        !capability && capability.reason() == Reason::DeviceLost;
    return result;
  }
  const VirtualBufferState *const first = virtual_input(*state);
  if (first == nullptr ||
      (mode == VirtualRunAdmissionMode::AsynchronousDeviceVsm &&
       first->count == 0u)) {
    result.status = Status::fail(Reason::PipelineInvalid);
    return result;
  }
  bool input_shape = true;
  for (std::size_t index = 0u; index < state->input_count; ++index) {
    const VirtualBufferState *const input = virtual_input(*state, index);
    input_shape =
        input_shape && input != nullptr && active_count <= input->count;
  }
  if (!input_shape ||
      ((state->geometry.route != VirtualRoute::Reduction &&
        state->geometry.route != VirtualRoute::GraphReduction) &&
       active_count > state->output->count)) {
    result.status = Status::fail(Reason::ShapeMismatch);
    return result;
  }
  result.claim = true;
  result.status = virtual_device_capability(*state);
  result.poison_pipeline =
      !result.status && result.status.reason() == Reason::DeviceLost;
  return result;
}

} // namespace rund::compute::detail

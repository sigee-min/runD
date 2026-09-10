#include "local.hpp"

namespace rund::compute::detail::virtual_run_dispatch {

VirtualRunDispatchResult
dispatch_accelerator(VirtualPipelineState &state, VirtualBacking &input,
                     VirtualBacking &output, const VirtualRunProjection &run,
                     const VirtualRunAdmission &admission, Stats &stats,
                     bool &handled) noexcept {
  handled = false;
  if (state.pipeline->device->backend == Backend::Cpu) {
    return {};
  }
  if (admission.try_recurrent) {
    VirtualExecutionSlidingPrepared sliding{};
    const Status sliding_ready =
        prepare_virtual_execution_sliding(state, run, sliding);
    if (sliding_ready) {
      const VirtualExecutionResult executed = execute_virtual_execution_sliding(
          state, input, output, run, sliding, stats);
      // Only the lowering boundary can authorize capability fallback. A
      // later unsupported status may follow native acceptance or a Final and
      // must remain terminal even when cleanup did not poison the pipeline.
      if (executed.disposition ==
              VirtualExecutionDisposition::CleanPreNativeDecline &&
          executed.status.reason() == Reason::BackendUnsupported &&
          !executed.poison_pipeline) {
        return {};
      }
      handled = true;
      return dispatch_result(executed);
    }
    if (sliding_ready.reason() != Reason::BackendUnsupported) {
      handled = true;
      return VirtualRunDispatchResult{
          .status = sliding_ready,
          .poison_pipeline = sliding_ready.reason() == Reason::DeviceLost,
          .direct_terminal = true,
      };
    }
  }

  VirtualExecutionWindowPrepared window{};
  const Status window_ready =
      admission.try_recurrent
          ? prepare_virtual_execution_window(state, run, window)
          : Status::fail(Reason::BackendUnsupported);
  if (window_ready) {
    handled = true;
    return dispatch_result(execute_virtual_execution_window(
        state, input, output, run, window, stats));
  }
  if (window_ready.reason() != Reason::BackendUnsupported) {
    handled = true;
    return VirtualRunDispatchResult{
        .status = window_ready,
        .poison_pipeline = window_ready.reason() == Reason::DeviceLost,
        .direct_terminal = true,
    };
  }

  VirtualExecutionPrepared execution{};
  const Status execution_ready =
      prepare_virtual_execution(state, run, execution);
  if (execution_ready) {
    handled = true;
    return dispatch_result(
        execute_virtual_execution(state, input, output, run, execution, stats));
  }
  if (execution_ready.reason() != Reason::BackendUnsupported) {
    handled = true;
    return VirtualRunDispatchResult{
        .status = execution_ready,
        .poison_pipeline = execution_ready.reason() == Reason::DeviceLost,
        .direct_terminal = true,
    };
  }
  return {};
}

} // namespace rund::compute::detail::virtual_run_dispatch

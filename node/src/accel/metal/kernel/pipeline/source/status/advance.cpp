#include "source.hpp"

namespace rund::node::accel::detail::metal_pipeline_status_source {

std::string_view advance() noexcept {
  return R"rundmetal(kernel void rund_pipeline_advance(
    device const uint *terminal0 [[buffer(0)]],
    device const uint *terminal1 [[buffer(1)]],
    device const uint *terminal2 [[buffer(2)]],
    device const uint *resident [[buffer(3)]],
    device ResidentState *states [[buffer(4)]],
    device PipelineControl *control [[buffer(5)]],
    constant WindowParams &params [[buffer(6)]],
    uint gid [[thread_position_in_grid]]) {
  if (gid != 0u) { return; }
  device ResidentState &state = states[params.state];
  uint active = 0u;
  const bool valid_phase = rund_pipeline_phase_valid(params.phase);
  if (!valid_phase) {
    if (control->reason == 0u) {
      control->reason = rund_pipeline_reason_invalid;
      control->failed_step = params.declared_step;
      control->failed_outer_window = 0xffffffffu;
      control->failed_inner_iteration = 0xffffffffu;
      control->failed_nested_phase = rund_pipeline_phase_none;
    }
    state.stopped = params.iteration + 1u;
    return;
  }
  if (params.phase == rund_pipeline_phase_fold) {
    if (state.stopped == 0u && params.inner_advance != 0u) {
      control->executed_inner_iteration_count =
          ulong(params.inner_advance) >
                  0xfffffffffffffffful -
                      control->executed_inner_iteration_count
              ? 0xfffffffffffffffful
              : control->executed_inner_iteration_count +
                    ulong(params.inner_advance);
    }
    active = state.stopped == 0u && control->reason == 0u ? 1u : 0u;
    if (active != 0u) {
      control->iteration_count =
          control->iteration_count == 0xfffffffffffffffful
              ? 0xfffffffffffffffful
              : control->iteration_count + 1ul;
      control->executed_outer_window_count =
          control->executed_outer_window_count == 0xfffffffffffffffful
              ? 0xfffffffffffffffful
              : control->executed_outer_window_count + 1ul;
    }
  } else if (params.phase == rund_pipeline_phase_action) {
    active = state.stopped == 0u && control->reason == 0u ? 1u : 0u;
    if (active != 0u) {
      control->executed_inner_iteration_count =
          ulong(params.inner_advance) >
                  0xfffffffffffffffful -
                      control->executed_inner_iteration_count
              ? 0xfffffffffffffffful
              : control->executed_inner_iteration_count +
                    ulong(params.inner_advance);
    }
  } else {
    // Fold completion is carried by the next Seed preflight. The final Fold
    // keeps its one-thread advance before canonicalization/publication.
    //
    // old: Seed(i) -> ... -> Fold(i) -> advance Fold(i) -> Seed(i+1)
    // new: Seed(i) -> ... -> Fold(i) -> Seed(i+1, completes Fold(i))
    if (params.phase == rund_pipeline_phase_seed && params.iteration != 0u &&
        state.stopped == 0u && control->reason == 0u) {
      control->executed_inner_iteration_count =
          ulong(params.inner_advance) >
                  0xfffffffffffffffful -
                      control->executed_inner_iteration_count
              ? 0xfffffffffffffffful
              : control->executed_inner_iteration_count +
                    ulong(params.inner_advance);
      control->iteration_count =
          control->iteration_count == 0xfffffffffffffffful
              ? 0xfffffffffffffffful
              : control->iteration_count + 1ul;
      control->executed_outer_window_count =
          control->executed_outer_window_count == 0xfffffffffffffffful
              ? 0xfffffffffffffffful
              : control->executed_outer_window_count + 1ul;
      state.current = 1u + ((params.iteration - 1u) & 1u);
    }
    const uint items = resident[params.count_offset_words];
    const ulong base = ulong(params.iteration) * ulong(params.tile);
    device const uint *terminal =
        state.current == 1u
            ? terminal1
            : (state.current == 2u ? terminal2 : terminal0);
    const bool ended =
        params.has_terminal != 0u &&
        terminal[params.terminal_offset_words[state.current]] ==
            params.expected;
    const bool overflow = state.stopped == 0u && items > params.maximum;
    if (overflow && control->reason == 0u) {
      control->reason = params.overflow_reason;
      control->failed_step = params.declared_step;
      control->overflow_ordinal = ulong(params.maximum);
      control->failed_outer_window =
          params.phase == rund_pipeline_phase_seed ? params.iteration
                                                    : 0xffffffffu;
      control->failed_inner_iteration = 0xffffffffu;
      control->failed_nested_phase =
          params.phase == rund_pipeline_phase_seed
              ? rund_pipeline_phase_seed
              : rund_pipeline_phase_none;
    }
    active = state.stopped == 0u && control->reason == 0u &&
                     base < ulong(items) && !ended
                 ? 1u
                 : 0u;
    if (params.phase == rund_pipeline_phase_seed && control->reason == 0u &&
        active == 0u) {
      control->skipped_iteration_count =
          control->skipped_iteration_count == 0xfffffffffffffffful
              ? 0xfffffffffffffffful
              : control->skipped_iteration_count + 1ul;
      control->skipped_outer_window_count =
          control->skipped_outer_window_count == 0xfffffffffffffffful
              ? 0xfffffffffffffffful
              : control->skipped_outer_window_count + 1ul;
      control->skipped_inner_iteration_count =
          ulong(params.inner_bound) >
                  0xfffffffffffffffful -
                      control->skipped_inner_iteration_count
              ? 0xfffffffffffffffful
              : control->skipped_inner_iteration_count +
                    ulong(params.inner_bound);
    }
  }
  if (active != 0u) {
    if (params.phase == rund_pipeline_phase_none ||
        params.phase == rund_pipeline_phase_fold) {
      state.current = 1u + (params.iteration & 1u);
    }
  } else if (state.stopped == 0u) {
    state.stopped = params.iteration + 1u;
  }
}

)rundmetal";
}

} // namespace rund::node::accel::detail::metal_pipeline_status_source

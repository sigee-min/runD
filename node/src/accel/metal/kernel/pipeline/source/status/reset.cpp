#include "source.hpp"

namespace rund::node::accel::detail::metal_pipeline_status_source {

std::string_view reset() noexcept {
  return R"rundmetal(inline void reset_telemetry(device PipelineControl *control) {
  control->generated_item_count = 0ul;
  control->generated_capacity = 0ul;
  control->indirect_dispatch_count = 0ul;
  control->indirect_work_item_count = 0ul;
  control->iteration_count = 0ul;
  control->skipped_iteration_count = 0ul;
  control->conflict_count = 0ul;
  control->overflow_ordinal = 0xfffffffffffffffful;
  control->failed_outer_window = 0xffffffffu;
  control->failed_inner_iteration = 0xffffffffu;
  control->failed_nested_phase = rund_pipeline_phase_none;
  control->reserved = 0u;
  control->executed_outer_window_count = 0ul;
  control->skipped_outer_window_count = 0ul;
  control->executed_inner_iteration_count = 0ul;
  control->skipped_inner_iteration_count = 0ul;
}

inline StepControl empty_step_control() {
  StepControl control;
  control.generated_item_count = 0ul;
  control.generated_capacity = 0ul;
  control.indirect_dispatch_count = 0ul;
  control.indirect_work_item_count = 0ul;
  control.iteration_count = 0ul;
  control.skipped_iteration_count = 0ul;
  control.conflict_count = 0ul;
  control.overflow_ordinal = 0xfffffffffffffffful;
  return control;
}

inline void reset_step_controls(device StepControl *steps, const uint count) {
  for (uint index = 0u; index < count; ++index) {
    steps[index] = empty_step_control();
  }
}

kernel void rund_pipeline_status_reset(
    device uint *raw [[buffer(0)]],
    constant ResetMeta *meta [[buffer(1)]],
    constant StatusParams &params [[buffer(2)]],
    uint gid [[thread_position_in_grid]]) {
  if (gid < params.reset_word_count) {
    // Reset ranges are a contiguous, strictly increasing partition of the
    // private raw prefix. Public words are overwritten by import dispatches.
    // Find the greatest range start not exceeding gid in O(log C), where C is
    // the immutable binding count.  Across R raw words this replaces the
    // prior O(R * C) scan with O(R * log C).
    uint low = 0u;
    uint high = params.reset_range_count;
    while (low + 1u < high) {
      const uint middle = low + ((high - low) >> 1u);
      if (meta[middle].raw_offset <= gid) {
        low = middle;
      } else {
        high = middle;
      }
    }
    raw[gid] = meta[low].reset;
  }
}

kernel void rund_pipeline_status_import(
    device const uint *source [[buffer(0)]],
    device uint *raw [[buffer(1)]],
    constant uint2 &range [[buffer(2)]],
    uint gid [[thread_position_in_grid]]) {
  if (gid < range.y) {
    raw[range.x + gid] = source[gid];
  }
}

kernel void rund_pipeline_status_complete(
    device PipelineControl *control [[buffer(0)]],
    constant StatusParams &params [[buffer(1)]],
    device ResidentState *states [[buffer(3)]],
    uint gid [[thread_position_in_grid]]) {
  if (params.phase == 0u) {
    if (gid < params.state_count) {
      states[gid] = ResidentState{0u, 0u};
    }
    if (gid == 0u) {
      control->reason = 0u;
      control->failed_step = 0xffffffffu;
      control->verified_prefix = 0u;
      reset_telemetry(control);
    }
    return;
  }
  if (gid == 0u) {
    control->generation += params.generation_stride;
    if (control->reason == 0u) {
      control->failed_step = 0xffffffffu;
      control->verified_prefix = params.declared_step_count;
    } else {
      control->verified_prefix = control->failed_step;
    }
  }
}

kernel void rund_pipeline_status_complete_profiled(
    device PipelineControl *control [[buffer(0)]],
    constant StatusParams &params [[buffer(1)]],
    device StepControl *steps [[buffer(2)]],
    device ResidentState *states [[buffer(3)]],
    uint gid [[thread_position_in_grid]]) {
  if (params.phase == 0u) {
    if (gid < params.state_count) {
      states[gid] = ResidentState{0u, 0u};
    }
    if (gid == 0u) {
      control->reason = 0u;
      control->failed_step = 0xffffffffu;
      control->verified_prefix = 0u;
      reset_telemetry(control);
      reset_step_controls(steps, params.declared_step_count);
    }
    return;
  }
  if (gid == 0u) {
    control->generation += params.generation_stride;
    if (control->reason == 0u) {
      control->failed_step = 0xffffffffu;
      control->verified_prefix = params.declared_step_count;
    } else {
      control->verified_prefix = control->failed_step;
    }
  }
}

)rundmetal";
}

} // namespace rund::node::accel::detail::metal_pipeline_status_source

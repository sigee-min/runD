#include "../source.hpp"

#include "../../../../kernel/backend/phase_source.hpp"

namespace rund::node::accel::detail {
namespace {

inline constexpr std::string_view MetalPipelineStatusPreamble = R"rundmetal(
#include <metal_stdlib>
using namespace metal;
)rundmetal";

inline constexpr std::string_view MetalPipelineStatusBody = R"rundmetal(
struct StatusSource {
  uint encoding;
  uint declared_step;
  uint policy0;
  uint policy1;
  uint policy2;
  uint policy3;
  uint limit_low;
  uint limit_high;
  uint raw_offset;
  uint telemetry;
  uint indirect_dispatch_count;
  uint work_item_count_low;
  uint work_item_count_high;
  uint failed_outer_window;
  uint failed_inner_iteration;
  uint failed_nested_phase;
};

struct StatusEntry {
  uint source;
  uint raw;
};

struct ResetMeta {
  uint raw_offset;
  uint reset;
};

struct StatusParams {
  uint reset_range_count;
  uint status_count;
  uint reset_word_count;
  uint declared_step_count;
  uint invalid_reason;
  uint generation_stride;
  uint source_count;
  uint phase;
  uint window_state;
  uint window_stop;
  uint window_inner_advance;
  uint state_count;
};

struct PipelineControl {
  uint generation;
  uint reason;
  uint failed_step;
  uint verified_prefix;
  ulong generated_item_count;
  ulong generated_capacity;
  ulong indirect_dispatch_count;
  ulong indirect_work_item_count;
  ulong iteration_count;
  ulong skipped_iteration_count;
  ulong conflict_count;
  ulong overflow_ordinal;
  uint failed_outer_window;
  uint failed_inner_iteration;
  uint failed_nested_phase;
  uint reserved;
  ulong executed_outer_window_count;
  ulong skipped_outer_window_count;
  ulong executed_inner_iteration_count;
  ulong skipped_inner_iteration_count;
};

struct PublishParams {
  ulong count;
  ulong source_offset_words[3];
  ulong source_stride_words[3];
  ulong target_offset_words;
  ulong target_stride_words;
  uint element_words;
  uint declared_step_count;
  uint state;
  uint final;
  uint stop;
  uint maximum;
  uint tile;
  uint outer;
  uint kind;
  ulong count_offset_words;
};

struct WindowParams {
  ulong count_offset_words;
  ulong terminal_offset_words[3];
  uint maximum;
  uint tile;
  uint iteration;
  uint expected;
  uint state;
  uint has_terminal;
  uint phase;
  uint declared_step;
  uint overflow_reason;
  uint inner_bound;
  uint inner_advance;
};

struct ResidentState {
  uint current;
  uint stopped;
};

static_assert(sizeof(ResidentState) == 2u * sizeof(uint),
              "ResidentState size must match the host ABI");
static_assert(alignof(ResidentState) == alignof(uint),
              "ResidentState alignment must match the host ABI");
static_assert(__builtin_offsetof(ResidentState, stopped) == 4u,
              "ResidentState::stopped offset must match the host ABI");

struct StepControl {
  ulong generated_item_count;
  ulong generated_capacity;
  ulong indirect_dispatch_count;
  ulong indirect_work_item_count;
  ulong iteration_count;
  ulong skipped_iteration_count;
  ulong conflict_count;
  ulong overflow_ordinal;
};

inline void reset_telemetry(device PipelineControl *control) {
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

kernel void rund_pipeline_publish(
    device const uint *seed [[buffer(0)]],
    device const uint *first [[buffer(1)]],
    device const uint *second [[buffer(2)]],
    device uint *target [[buffer(3)]],
    device const PipelineControl *control [[buffer(4)]],
    device const ResidentState *states [[buffer(5)]],
    constant PublishParams &params [[buffer(6)]],
    device const uint *resident [[buffer(7)]],
    uint gid [[thread_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]]) {
  threadgroup uint publish;
  if (tid == 0u) {
    const ResidentState state = states[params.state];
    if (params.kind == 1u) {
      const ulong base = ulong(params.outer) * ulong(params.tile);
      publish = control->reason == 0u && state.stopped == 0u &&
                        base < min(ulong(resident[params.count_offset_words]),
                                   ulong(params.maximum))
                    ? 1u
                    : 0u;
    } else {
      publish = params.stop == 0u
                    ? (control->reason == 0u &&
                       control->failed_step == 0xffffffffu &&
                       control->verified_prefix == params.declared_step_count
                   ? 1u
                   : 0u)
                    : (control->reason == 0u &&
                               state.current != params.final
                           ? 1u
                           : 0u);
    }
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  const ulong base = ulong(params.outer) * ulong(params.tile);
  ulong active_count = params.count;
  if (params.kind == 1u && publish != 0u) {
    active_count = min(
        min(ulong(params.tile), ulong(params.maximum) - base),
        ulong(resident[params.count_offset_words]) - base);
  }
  if (publish == 0u || ulong(gid) >= active_count) { return; }
  const uint current = params.kind == 1u
                           ? 0u
                           : (params.stop == 0u
                                  ? params.final
                                  : states[params.state].current);
  device const uint *source =
      current == 1u ? first : (current == 2u ? second : seed);
  const ulong source_word =
      params.source_offset_words[current] +
      ulong(gid) * params.source_stride_words[current];
  const ulong target_word =
      params.target_offset_words +
      (params.kind == 1u ? base + ulong(gid) : ulong(gid)) *
          params.target_stride_words;
  target[target_word] = source[source_word];
  if (params.element_words == 2u) {
    target[target_word + 1u] = source[source_word + 1u];
  }
}

kernel void rund_pipeline_advance(
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

inline uint policy_reason(const uint policy) {
  return policy & 0xffffu;
}

inline uint policy_priority(const uint policy) {
  return policy >> 16u;
}

inline uint select_reason(const StatusSource item, const uint raw) {
  if (item.encoding == 0u) {
    return raw == 0u ? 0u : policy_reason(item.policy0);
  }
  if (item.encoding == 1u || item.encoding == 3u) {
    const uint reason0 = policy_reason(item.policy0);
    const uint reason1 = policy_reason(item.policy1);
    const uint reason2 = policy_reason(item.policy2);
    const uint known = (reason0 == 0u ? 0u : 1u) |
                       (reason1 == 0u ? 0u : 2u) |
                       (reason2 == 0u ? 0u : 4u);
    if ((raw & ~known) != 0u) {
      return 0xffffffffu;
    }
    uint selected = 0u;
    uint priority = 0xffffffffu;
    if ((raw & 1u) != 0u && policy_priority(item.policy0) < priority) {
      selected = reason0;
      priority = policy_priority(item.policy0);
    }
    if ((raw & 2u) != 0u && policy_priority(item.policy1) < priority) {
      selected = reason1;
      priority = policy_priority(item.policy1);
    }
    if ((raw & 4u) != 0u && policy_priority(item.policy2) < priority) {
      selected = reason2;
    }
    return selected;
  }
  if (item.encoding == 2u) {
    return raw == 0u
               ? 0u
               : (raw == 1u
                      ? policy_reason(item.policy0)
                      : (raw == 2u ? policy_reason(item.policy1)
                                   : 0xffffffffu));
  }
  if (item.encoding == 4u) {
    return raw == 0xffffffffu ? 0u : policy_reason(item.policy0);
  }
  if (item.encoding == 5u) {
    return raw == 0xffffffffu
               ? 0u
               : ((raw & 1u) == 0u ? policy_reason(item.policy0)
                                    : policy_reason(item.policy1));
  }
  if (item.encoding == 6u) {
    const ulong limit =
        (ulong(item.limit_high) << 32u) | ulong(item.limit_low);
    return ulong(raw) > limit ? policy_reason(item.policy0) : 0u;
  }
  if (item.encoding == 7u) {
    if (raw == 0u) {
      return 0u;
    }
    if (raw == 1u) {
      return policy_reason(item.policy0);
    }
    if (raw == 2u) {
      return policy_reason(item.policy1);
    }
    if (raw == 3u) {
      return policy_reason(item.policy2);
    }
    if (raw == 4u) {
      return policy_reason(item.policy3);
    }
    return 0xffffffffu;
  }
  return 0xffffffffu;
}

inline ulong status_add(const ulong left, const ulong right) {
  return right > 0xfffffffffffffffful - left
             ? 0xfffffffffffffffful
             : left + right;
}

inline StepControl merge_step_control(const StepControl left,
                                      const StepControl right) {
  StepControl merged;
  merged.generated_item_count =
      status_add(left.generated_item_count, right.generated_item_count);
  merged.generated_capacity =
      status_add(left.generated_capacity, right.generated_capacity);
  merged.indirect_dispatch_count =
      status_add(left.indirect_dispatch_count,
                 right.indirect_dispatch_count);
  merged.indirect_work_item_count =
      status_add(left.indirect_work_item_count,
                 right.indirect_work_item_count);
  merged.iteration_count =
      status_add(left.iteration_count, right.iteration_count);
  merged.skipped_iteration_count =
      status_add(left.skipped_iteration_count,
                 right.skipped_iteration_count);
  merged.conflict_count =
      status_add(left.conflict_count, right.conflict_count);
  merged.overflow_ordinal =
      min(left.overflow_ordinal, right.overflow_ordinal);
  return merged;
}

inline StepControl status_step_control(const StatusSource item,
                                       device const uint *raw) {
  StepControl result = empty_step_control();
  if (item.telemetry == 0u) {
    return result;
  }
  const uint status = raw[item.raw_offset];
  const ulong logical = ulong(raw[item.raw_offset + 1u]);
  if (item.telemetry == 2u) {
    result.conflict_count = ulong(raw[item.raw_offset + 2u]);
  }
  if (status != 0u) {
    result.overflow_ordinal = logical;
    return result;
  }
  result.generated_item_count = logical;
  result.generated_capacity =
      (ulong(item.limit_high) << 32u) | ulong(item.limit_low);
  result.indirect_dispatch_count = ulong(item.indirect_dispatch_count);
  result.indirect_work_item_count = logical;
  if (item.telemetry == 2u) {
    result.indirect_work_item_count = status_add(
        result.indirect_work_item_count,
        (ulong(item.work_item_count_high) << 32u) |
            ulong(item.work_item_count_low));
  }
  return result;
}

inline void store_pipeline_telemetry(device PipelineControl *control,
                                     const StepControl value) {
  control->generated_item_count = value.generated_item_count;
  control->generated_capacity = value.generated_capacity;
  control->indirect_dispatch_count = value.indirect_dispatch_count;
  control->indirect_work_item_count = value.indirect_work_item_count;
  control->iteration_count = value.iteration_count;
  control->skipped_iteration_count = value.skipped_iteration_count;
  control->conflict_count = value.conflict_count;
  control->overflow_ordinal = value.overflow_ordinal;
}

inline void merge_pipeline_telemetry(device PipelineControl *control,
                                     const StepControl value) {
  StepControl current;
  current.generated_item_count = control->generated_item_count;
  current.generated_capacity = control->generated_capacity;
  current.indirect_dispatch_count = control->indirect_dispatch_count;
  current.indirect_work_item_count = control->indirect_work_item_count;
  current.iteration_count = control->iteration_count;
  current.skipped_iteration_count = control->skipped_iteration_count;
  current.conflict_count = control->conflict_count;
  current.overflow_ordinal = control->overflow_ordinal;
  store_pipeline_telemetry(control, merge_step_control(current, value));
}

inline void close_failed_nested_window(
    device PipelineControl *control,
    device ResidentState *states,
    constant StatusParams &params) {
  if (params.window_state == 0xffffffffu || control->reason == 0u) {
    return;
  }
  device ResidentState &state = states[params.window_state];
  if (state.stopped == 0u) {
    control->executed_inner_iteration_count =
        ulong(params.window_inner_advance) >
                0xfffffffffffffffful -
                    control->executed_inner_iteration_count
            ? 0xfffffffffffffffful
            : control->executed_inner_iteration_count +
                  ulong(params.window_inner_advance);
    state.stopped = params.window_stop;
  }
}

kernel void rund_pipeline_status_reduce(
    device const uint *raw [[buffer(0)]],
    device PipelineControl *control [[buffer(1)]],
    constant StatusEntry *entries [[buffer(2)]],
    constant StatusSource *sources [[buffer(3)]],
    constant StatusParams &params [[buffer(4)]],
    device ResidentState *states [[buffer(5)]],
    uint tid [[thread_index_in_threadgroup]]) {
  if (control->reason != 0u) {
    return;
  }
  threadgroup uint keys[128];
  threadgroup uint reasons[128];
  uint key = 0xffffffffu;
  uint selected = 0u;
  for (uint index = tid; index < params.status_count; index += 128u) {
    const StatusEntry entry = entries[index];
    const StatusSource item = sources[entry.source];
    const uint value = raw[entry.raw];
    uint reason = select_reason(item, value);
    if (reason == 0xffffffffu) {
      reason = params.invalid_reason;
    }
    if (reason != 0u && index < key) {
      key = index;
      selected = reason;
    }
  }
  keys[tid] = key;
  reasons[tid] = selected;
  threadgroup_barrier(mem_flags::mem_threadgroup);
  for (uint stride = 64u; stride != 0u; stride >>= 1u) {
    if (tid < stride && keys[tid + stride] < keys[tid]) {
      keys[tid] = keys[tid + stride];
      reasons[tid] = reasons[tid + stride];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }
  if (tid == 0u) {
    const uint failed = keys[0];
    if (failed != 0xffffffffu) {
      control->reason = reasons[0];
      const StatusSource source = sources[entries[failed].source];
      control->failed_step = source.declared_step;
      control->failed_outer_window = source.failed_outer_window;
      control->failed_inner_iteration = source.failed_inner_iteration;
      control->failed_nested_phase = source.failed_nested_phase;
    }
    ulong generated_item_count = 0ul;
    ulong generated_capacity = 0ul;
    ulong indirect_dispatch_count = 0ul;
    ulong indirect_work_item_count = 0ul;
    ulong conflict_count = 0ul;
    ulong overflow_ordinal = 0xfffffffffffffffful;
    for (uint source_index = 0u; source_index < params.source_count;
         ++source_index) {
      const StatusSource item = sources[source_index];
      if (item.telemetry == 0u) { continue; }
      const uint status = raw[item.raw_offset];
      const ulong logical = ulong(raw[item.raw_offset + 1u]);
      if (item.telemetry == 2u) {
        conflict_count = status_add(
            conflict_count, ulong(raw[item.raw_offset + 2u]));
      }
      if (status != 0u) {
        overflow_ordinal = min(overflow_ordinal, logical);
        continue;
      }
      const ulong capacity =
          (ulong(item.limit_high) << 32u) | ulong(item.limit_low);
      generated_item_count = status_add(generated_item_count, logical);
      generated_capacity = status_add(generated_capacity, capacity);
      indirect_dispatch_count = status_add(
          indirect_dispatch_count, ulong(item.indirect_dispatch_count));
      indirect_work_item_count =
          status_add(indirect_work_item_count, logical);
      if (item.telemetry == 2u) {
        const ulong work_item_count =
            (ulong(item.work_item_count_high) << 32u) |
            ulong(item.work_item_count_low);
        indirect_work_item_count =
            status_add(indirect_work_item_count, work_item_count);
      }
    }
    StepControl total = empty_step_control();
    total.generated_item_count = generated_item_count;
    total.generated_capacity = generated_capacity;
    total.indirect_dispatch_count = indirect_dispatch_count;
    total.indirect_work_item_count = indirect_work_item_count;
    total.conflict_count = conflict_count;
    total.overflow_ordinal = overflow_ordinal;
    merge_pipeline_telemetry(control, total);
    close_failed_nested_window(control, states, params);
  }
}

kernel void rund_pipeline_status_reduce_profiled(
    device const uint *raw [[buffer(0)]],
    device PipelineControl *control [[buffer(1)]],
    constant StatusEntry *entries [[buffer(2)]],
    constant StatusSource *sources [[buffer(3)]],
    constant StatusParams &params [[buffer(4)]],
    device ResidentState *states [[buffer(5)]],
    device StepControl *steps [[buffer(6)]],
    uint tid [[thread_index_in_threadgroup]]) {
  if (control->reason != 0u) {
    return;
  }
  threadgroup uint keys[128];
  threadgroup uint reasons[128];
  uint key = 0xffffffffu;
  uint selected = 0u;
  for (uint index = tid; index < params.status_count; index += 128u) {
    const StatusEntry entry = entries[index];
    const StatusSource item = sources[entry.source];
    const uint value = raw[entry.raw];
    uint reason = select_reason(item, value);
    if (reason == 0xffffffffu) {
      reason = params.invalid_reason;
    }
    if (reason != 0u && index < key) {
      key = index;
      selected = reason;
    }
  }
  keys[tid] = key;
  reasons[tid] = selected;
  threadgroup_barrier(mem_flags::mem_threadgroup);
  for (uint stride = 64u; stride != 0u; stride >>= 1u) {
    if (tid < stride && keys[tid + stride] < keys[tid]) {
      keys[tid] = keys[tid + stride];
      reasons[tid] = reasons[tid + stride];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }
  if (tid == 0u) {
    const uint failed = keys[0];
    if (failed != 0xffffffffu) {
      control->reason = reasons[0];
      const StatusSource source = sources[entries[failed].source];
      control->failed_step = source.declared_step;
      control->failed_outer_window = source.failed_outer_window;
      control->failed_inner_iteration = source.failed_inner_iteration;
      control->failed_nested_phase = source.failed_nested_phase;
    }
    StepControl total = empty_step_control();
    for (uint source_index = 0u; source_index < params.source_count;
         ++source_index) {
      const StatusSource item = sources[source_index];
      const StepControl value = status_step_control(item, raw);
      total = merge_step_control(total, value);
      if (item.telemetry != 0u &&
          item.declared_step < params.declared_step_count) {
        steps[item.declared_step] =
            merge_step_control(steps[item.declared_step], value);
      }
    }
    merge_pipeline_telemetry(control, total);
    close_failed_nested_window(control, states, params);
  }
}

)rundmetal";

template <typename Sink>
[[nodiscard]] bool EmitMetalPipelineStatusSource(Sink &sink) noexcept(
    noexcept(sink.append(std::string_view{}))) {
  return sink.append(MetalPipelineStatusPreamble) &&
         EmitPipelineNestedPhaseContract(
             sink, PipelineNestedPhaseSourceLanguage::Metal) &&
         sink.append(MetalPipelineStatusBody);
}

} // namespace

std::string_view MetalPipelineStatusSource() noexcept {
  static const auto source = backend_source_recipe::materialize_fixed<
      MetalPipelineStatusPreamble.size() + MetalPipelineStatusBody.size() +
      1024u>(
      [](auto &sink) noexcept { return EmitMetalPipelineStatusSource(sink); });
  return source.text();
}

} // namespace rund::node::accel::detail

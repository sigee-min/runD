#include "source.hpp"

namespace rund::node::accel::detail::metal_pipeline_status_source {

std::string_view reduce() noexcept {
  return R"rundmetal(inline uint policy_reason(const uint policy) {
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
}

} // namespace rund::node::accel::detail::metal_pipeline_status_source

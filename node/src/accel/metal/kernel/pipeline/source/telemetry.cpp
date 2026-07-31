#include "../source.hpp"

namespace rund::node::accel::detail {

std::string_view MetalPipelineTelemetrySourceText() noexcept {
  return R"rundmetal(struct TelemetryParams {
  uint kind;
  uint primary_word_count;
  uint count_source;
  uint predicate_source;
  uint has_count;
  uint has_predicate;
  uint iteration;
  uint count_word_offset;
  uint predicate_word_offset;
  uint indirect_dispatch_count;
  uint declared_step_count;
  uint declared_step;
  ulong capacity;
  ulong predicate_expected;
  ulong work_item_count;
};

inline ulong telemetry_scalar(device const uint *words, const uint source) {
  return source == 2u ? ulong(words[0]) | (ulong(words[1]) << 32u)
                      : ulong(words[0]);
}

inline ulong telemetry_add(const ulong left, const ulong right) {
  return right > 0xfffffffffffffffful - left
             ? 0xfffffffffffffffful
             : left + right;
}

inline ulong telemetry_multiply_u32(const ulong left, const uint right) {
  const ulong low_product = ulong(uint(left)) * ulong(right);
  const ulong high_product = ulong(uint(left >> 32u)) * ulong(right);
  if (high_product > 0xfffffffful) {
    return 0xfffffffffffffffful;
  }
  const ulong upper = high_product << 32u;
  return low_product > 0xfffffffffffffffful - upper
             ? 0xfffffffffffffffful
             : upper + low_product;
}

inline StepControl telemetry_step_control(
    device const uint *primary, device const uint *count_words,
    device const uint *predicate_words, constant TelemetryParams &params) {
  StepControl result = empty_step_control();
  if (params.kind == 2u) {
    if (params.primary_word_count >= 4u) {
      result.conflict_count = ulong(primary[2]);
      if (primary[0] != 0u) {
        result.overflow_ordinal = ulong(primary[1]);
      } else {
        const ulong logical = ulong(primary[1]);
        result.generated_item_count = logical;
        result.generated_capacity = params.capacity;
        result.indirect_dispatch_count =
            ulong(params.indirect_dispatch_count);
        result.indirect_work_item_count =
            telemetry_add(logical, params.work_item_count);
      }
    }
    return result;
  }
  if (params.kind == 4u) {
    if (primary[0] != 0u) {
      result.overflow_ordinal = ulong(primary[1]);
    } else {
      const ulong logical = ulong(primary[1]);
      result.generated_item_count = logical;
      result.generated_capacity = params.capacity;
      result.indirect_dispatch_count =
          ulong(params.indirect_dispatch_count);
      result.indirect_work_item_count = logical;
    }
    return result;
  }
  if (params.kind == 3u) {
    const ulong logical = telemetry_scalar(count_words +
                                               params.count_word_offset,
                                           params.count_source);
    result.generated_item_count = logical;
    result.generated_capacity = params.capacity;
    if (logical > params.capacity) {
      result.overflow_ordinal = params.capacity;
      return result;
    }
    if (params.iteration != 0u) {
      if (logical == 0ul) {
        result.skipped_iteration_count = 1ul;
        return result;
      }
      result.iteration_count = 1ul;
    }
    result.indirect_dispatch_count =
        ulong(params.indirect_dispatch_count);
    result.indirect_work_item_count =
        telemetry_multiply_u32(logical, params.indirect_dispatch_count);
    return result;
  }
  if (params.kind != 1u || params.primary_word_count == 0u ||
      (params.primary_word_count & 3u) != 0u) {
    return result;
  }
  const ulong logical = params.has_count != 0u
                            ? telemetry_scalar(count_words +
                                                   params.count_word_offset,
                                               params.count_source)
                            : params.capacity;
  if (params.has_predicate != 0u) {
    const ulong predicate =
        telemetry_scalar(predicate_words + params.predicate_word_offset,
                         params.predicate_source);
    if (predicate != params.predicate_expected) {
      result.skipped_iteration_count = 1ul;
      return result;
    }
    result.iteration_count = 1ul;
  }
  if (params.has_count == 0u) {
    return result;
  }
  result.generated_item_count = logical;
  result.generated_capacity = params.capacity;
  if (logical > params.capacity) {
    result.overflow_ordinal = params.capacity;
    return result;
  }
  if (params.iteration != 0u && params.has_predicate == 0u) {
    if (logical == 0ul) {
      result.skipped_iteration_count = 1ul;
      return result;
    }
    result.iteration_count = 1ul;
  }
  result.indirect_dispatch_count = ulong(params.indirect_dispatch_count);
  ulong active = 0ul;
  for (uint index = 3u; index < params.primary_word_count; index += 4u) {
    active = telemetry_add(active, ulong(primary[index]));
  }
  result.indirect_work_item_count = active;
  return result;
}

kernel void rund_pipeline_telemetry_accumulate(
    device const uint *primary [[buffer(0)]],
    device const uint *count_words [[buffer(1)]],
    device const uint *predicate_words [[buffer(2)]],
    device PipelineControl *control [[buffer(3)]],
    constant TelemetryParams &params [[buffer(4)]],
    uint gid [[thread_position_in_grid]]) {
  if (gid != 0u) { return; }
  if (control->reason != 0u &&
      control->failed_step != params.declared_step) {
    return;
  }
  if (params.kind == 2u) {
    if (params.primary_word_count >= 4u) {
      control->conflict_count =
          telemetry_add(control->conflict_count, ulong(primary[2]));
      if (primary[0] != 0u) {
        control->overflow_ordinal =
            min(control->overflow_ordinal, ulong(primary[1]));
      } else {
        const ulong logical = ulong(primary[1]);
        control->generated_item_count =
            telemetry_add(control->generated_item_count, logical);
        control->generated_capacity =
            telemetry_add(control->generated_capacity, params.capacity);
        control->indirect_dispatch_count = telemetry_add(
            control->indirect_dispatch_count,
            ulong(params.indirect_dispatch_count));
        control->indirect_work_item_count = telemetry_add(
            control->indirect_work_item_count,
            telemetry_add(logical, params.work_item_count));
      }
    }
    return;
  }
  if (params.kind == 4u) {
    if (primary[0] != 0u) {
      control->overflow_ordinal =
          min(control->overflow_ordinal, ulong(primary[1]));
    } else {
      const ulong logical = ulong(primary[1]);
      control->generated_item_count =
          telemetry_add(control->generated_item_count, logical);
      control->generated_capacity =
          telemetry_add(control->generated_capacity, params.capacity);
      control->indirect_dispatch_count = telemetry_add(
          control->indirect_dispatch_count,
          ulong(params.indirect_dispatch_count));
      control->indirect_work_item_count = telemetry_add(
          control->indirect_work_item_count, logical);
    }
    return;
  }
  if (params.kind == 3u) {
    const ulong logical = telemetry_scalar(count_words +
                                               params.count_word_offset,
                                           params.count_source);
    control->generated_item_count =
        telemetry_add(control->generated_item_count, logical);
    control->generated_capacity =
        telemetry_add(control->generated_capacity, params.capacity);
    if (logical > params.capacity) {
      control->overflow_ordinal =
          min(control->overflow_ordinal, params.capacity);
      return;
    }
    if (params.iteration != 0u) {
      if (logical == 0ul) {
        control->skipped_iteration_count =
            telemetry_add(control->skipped_iteration_count, 1ul);
        return;
      }
      control->iteration_count =
          telemetry_add(control->iteration_count, 1ul);
    }
    control->indirect_dispatch_count = telemetry_add(
        control->indirect_dispatch_count,
        ulong(params.indirect_dispatch_count));
    control->indirect_work_item_count = telemetry_add(
        control->indirect_work_item_count,
        telemetry_multiply_u32(logical, params.indirect_dispatch_count));
    return;
  }
  if (params.kind != 1u || params.primary_word_count == 0u ||
      (params.primary_word_count & 3u) != 0u) { return; }
  const ulong logical = params.has_count != 0u
                            ? telemetry_scalar(count_words +
                                                   params.count_word_offset,
                                               params.count_source)
                            : params.capacity;
  if (params.has_predicate != 0u) {
    const ulong predicate =
        telemetry_scalar(predicate_words + params.predicate_word_offset,
                         params.predicate_source);
    if (predicate != params.predicate_expected) {
      control->skipped_iteration_count =
          telemetry_add(control->skipped_iteration_count, 1ul);
      return;
    }
    control->iteration_count = telemetry_add(control->iteration_count, 1ul);
  }
  if (params.has_count == 0u) { return; }
  control->generated_item_count =
      telemetry_add(control->generated_item_count, logical);
  control->generated_capacity =
      telemetry_add(control->generated_capacity, params.capacity);
  if (logical > params.capacity) {
    control->overflow_ordinal =
        min(control->overflow_ordinal, params.capacity);
    return;
  }
  if (params.iteration != 0u && params.has_predicate == 0u) {
    if (logical == 0ul) {
      control->skipped_iteration_count =
          telemetry_add(control->skipped_iteration_count, 1ul);
      return;
    }
    control->iteration_count = telemetry_add(control->iteration_count, 1ul);
  }
  control->indirect_dispatch_count = telemetry_add(
      control->indirect_dispatch_count,
      ulong(params.indirect_dispatch_count));
  ulong active = 0ul;
  for (uint index = 3u; index < params.primary_word_count; index += 4u) {
    active = telemetry_add(active, ulong(primary[index]));
  }
  control->indirect_work_item_count =
      telemetry_add(control->indirect_work_item_count, active);
}

kernel void rund_pipeline_telemetry_accumulate_profiled(
    device const uint *primary [[buffer(0)]],
    device const uint *count_words [[buffer(1)]],
    device const uint *predicate_words [[buffer(2)]],
    device PipelineControl *control [[buffer(3)]],
    constant TelemetryParams &params [[buffer(4)]],
    device StepControl *steps [[buffer(5)]],
    constant uint &declared_step [[buffer(6)]],
    uint gid [[thread_position_in_grid]]) {
  if (gid != 0u) {
    return;
  }
  if (control->reason != 0u &&
      control->failed_step != declared_step) {
    return;
  }
  const StepControl value =
      telemetry_step_control(primary, count_words, predicate_words, params);
  merge_pipeline_telemetry(control, value);
  steps[declared_step] =
      merge_step_control(steps[declared_step], value);
}

)rundmetal";
}

} // namespace rund::node::accel::detail

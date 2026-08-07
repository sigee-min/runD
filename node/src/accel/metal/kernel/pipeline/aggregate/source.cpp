#include "source.hpp"

#include "../../../../kernel/backend/phase_source.hpp"

namespace rund::node::accel::detail {
namespace {

inline constexpr std::string_view MetalNestedAggregatePreamble = R"rundmetal(
#include <metal_stdlib>
using namespace metal;
)rundmetal";

inline constexpr std::string_view MetalNestedAggregateBody = R"rundmetal(
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

struct ScalarExpr {
  uint lhs;
  uint rhs;
  uint immediate;
  uint reserved;
};

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

struct AggregateParams {
  ulong queue_offset_words;
  ulong queue_stride_words;
  ulong domain_offset_words;
  ulong domain_stride_words;
  ulong count_offset_words;
  ulong seed_offset_words;
  ulong target_offset_words;
  ulong tile_low_offset_words;
  ulong tile_status_offset_words;
  ulong queue_count;
  ulong domain_count;
  uint maximum;
  uint tile;
  uint outer_bound;
  uint inner_bound;
  uint generation_stride;
  uint declared_step_count;
  uint declared_step;
  uint count_overflow_reason;
  uint gather_reason;
  uint reduce_reason;
  uint profile_steps;
  uint profile_count;
  uint profile_seed_first;
  uint reserved;
  ScalarExpr action;
  ScalarExpr fold;
};

static_assert(sizeof(PipelineControl) == 128u,
              "PipelineControl must match the host ABI");
static_assert(sizeof(ScalarExpr) == 16u,
              "ScalarExpr must match the host ABI");
static_assert(sizeof(StepControl) == 64u,
              "StepControl must match the host ABI");
static_assert(sizeof(AggregateParams) == 176u,
              "AggregateParams must match the host ABI");

inline ulong saturated_add(const ulong lhs, const ulong rhs) {
  return rhs > 0xfffffffffffffffful - lhs ? 0xfffffffffffffffful
                                          : lhs + rhs;
}

inline uint2 wide_add(const uint2 lhs, const uint2 rhs) {
  // Common admission bounds the term count and every term to U32, so the
  // exact mathematical sum is strictly smaller than 2^64. uint2 therefore
  // carries the complete proof domain without a hidden saturation point.
  const uint low = lhs.x + rhs.x;
  return uint2(low, lhs.y + rhs.y + (low < lhs.x ? 1u : 0u));
}

inline uint2 simd_wide_sum(uint2 value, const uint lane,
                           const uint simd_width) {
  for (uint offset = simd_width >> 1u; offset != 0u; offset >>= 1u) {
    const uint2 other = uint2(simd_shuffle_down(value.x, offset),
                              simd_shuffle_down(value.y, offset));
    if (lane < offset) {
      value = wide_add(value, other);
    }
  }
  return value;
}

inline uint scalar_value(const uint source, const uint tile_state,
                         const uint tile_count, const uint outer_state,
                         const uint immediate) {
  if (source == 0u) { return tile_state; }
  if (source == 1u) { return tile_count; }
  if (source == 2u) { return outer_state; }
  return immediate;
}

inline uint evaluate_add(const ScalarExpr expr, const uint tile_state,
                         const uint tile_count, const uint outer_state) {
  return scalar_value(expr.lhs, tile_state, tile_count, outer_state,
                      expr.immediate) +
         scalar_value(expr.rhs, tile_state, tile_count, outer_state,
                      expr.immediate);
}

inline void reset_control(device PipelineControl *control) {
  control->reason = 0u;
  control->failed_step = 0xffffffffu;
  control->verified_prefix = 0u;
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

inline StepControl bounded_telemetry(const uint live, const uint tile,
                                     const uint operations) {
  StepControl result = empty_step_control();
  result.generated_item_count = ulong(live) * ulong(operations);
  result.generated_capacity = ulong(tile) * ulong(operations);
  result.indirect_dispatch_count = ulong(operations);
  result.indirect_work_item_count = ulong(live) * ulong(operations);
  return result;
}

inline void store_bounded_telemetry(device PipelineControl *control,
                                    const StepControl value) {
  control->generated_item_count = value.generated_item_count;
  control->generated_capacity = value.generated_capacity;
  control->indirect_dispatch_count = value.indirect_dispatch_count;
  control->indirect_work_item_count = value.indirect_work_item_count;
}

inline StepControl bounded_prefix_telemetry(const uint outer,
                                            const uint live,
                                            const uint tile,
                                            const uint operations) {
  StepControl result = empty_step_control();
  const ulong completed = ulong(outer) * ulong(tile);
  result.generated_item_count = 2ul * completed +
                                ulong(operations) * ulong(live);
  result.generated_capacity =
      (2ul * ulong(outer) + ulong(operations)) * ulong(tile);
  result.indirect_dispatch_count =
      2ul * ulong(outer) + ulong(operations);
  result.indirect_work_item_count = result.generated_item_count;
  return result;
}

inline void fail_seed(device PipelineControl *control, const uint reason,
                      const uint declared_step, const uint outer) {
  control->reason = reason;
  control->failed_step = declared_step;
  control->failed_outer_window = outer;
  control->failed_inner_iteration = 0xffffffffu;
  control->failed_nested_phase = rund_pipeline_phase_seed;
}

kernel void rund_pipeline_nested_aggregate_reduce_u32(
    device const uint *queue [[buffer(0)]],
    device const uint *domain [[buffer(1)]],
    device const uint *count [[buffer(2)]],
    constant AggregateParams &params [[buffer(3)]],
    device uint *tile_low [[buffer(4)]],
    device uint *tile_status [[buffer(5)]],
    uint tid [[thread_index_in_threadgroup]],
    uint width [[threads_per_threadgroup]], uint group
    [[threadgroup_position_in_grid]],
    uint lane [[thread_index_in_simdgroup]],
    uint simd_group [[simdgroup_index_in_threadgroup]],
    uint simd_groups [[simdgroups_per_threadgroup]],
    uint simd_width [[threads_per_simdgroup]]) {
  threadgroup uint partial_low[32];
  threadgroup uint partial_high[32];
  threadgroup uint partial_bad[32];
  const uint outer = group;
  const uint items = count[params.count_offset_words];
  const bool count_valid = items <= params.maximum;
  const uint active_outer =
      count_valid
          ? uint((ulong(items) + ulong(params.tile) - 1ul) /
                 ulong(params.tile))
          : 0u;
  const bool active = outer < active_outer;
  // `active` is uniform for the complete threadgroup. Inactive or invalid
  // tails publish no row and may leave before any threadgroup barrier because
  // finalize derives and reads only the active prefix from the same count.
  if (!active) { return; }
  const ulong base = ulong(outer) * ulong(params.tile);
  const uint live =
      uint(min(ulong(params.tile), ulong(items) - base));
  uint2 local_sum = uint2(0u);
  uint local_bad = 0xffffffffu;
  for (ulong offset = ulong(tid); offset < ulong(live);
       offset += ulong(width)) {
    const ulong ordinal = base + offset;
    if (ordinal >= params.queue_count) {
      local_bad = min(local_bad, uint(offset));
      continue;
    }
    const uint item =
        queue[params.queue_offset_words +
              ordinal * params.queue_stride_words];
    if (ulong(item) >= params.domain_count) {
      local_bad = min(local_bad, uint(offset));
    } else {
      const uint value =
          domain[params.domain_offset_words +
                 ulong(item) * params.domain_stride_words];
      local_sum = wide_add(local_sum, uint2(value, 0u));
    }
  }
  const uint2 simd_total = simd_wide_sum(local_sum, lane, simd_width);
  const uint simd_bad = simd_min(local_bad);
  if (lane == 0u) {
    partial_low[simd_group] = simd_total.x;
    partial_high[simd_group] = simd_total.y;
    partial_bad[simd_group] = simd_bad;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (simd_group == 0u) {
    uint2 total = lane < simd_groups
                      ? uint2(partial_low[lane], partial_high[lane])
                      : uint2(0u);
    uint bad = lane < simd_groups ? partial_bad[lane] : 0xffffffffu;
    total = simd_wide_sum(total, lane, simd_width);
    bad = simd_min(bad);
    if (lane == 0u) {
      partial_low[0] = total.x;
      partial_high[0] = total.y;
      partial_bad[0] = bad;
    }
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (tid == 0u) {
    const uint bad = partial_bad[0];
    tile_low[params.tile_low_offset_words + ulong(outer)] = partial_low[0];
    // Invalid offsets are strictly less than tile. `tile` is the overflow
    // marker and UINT_MAX is success; common/native admission excludes
    // tile==UINT_MAX, so all three states are disjoint in one U32 word.
    tile_status[params.tile_status_offset_words + ulong(outer)] =
        bad != 0xffffffffu
            ? bad
            : (partial_high[0] != 0u ? params.tile : 0xffffffffu);
  }
}

kernel void rund_pipeline_nested_aggregate_finalize_u32(
    device const uint *count [[buffer(0)]],
    device const uint *seed [[buffer(1)]],
    device uint *target [[buffer(2)]],
    device PipelineControl *control [[buffer(3)]],
    constant AggregateParams &params [[buffer(4)]],
    device StepControl *steps [[buffer(5)]],
    device const uint *tile_low [[buffer(6)]],
    device const uint *tile_status [[buffer(7)]]) {
  reset_control(control);
  if (params.profile_steps != 0u) {
    for (uint index = 0u; index < params.profile_count; ++index) {
      steps[index] = empty_step_control();
    }
  }

  const uint items = count[params.count_offset_words];
  bool alive = items <= params.maximum;
  const uint active_outer =
      alive ? uint((ulong(items) + ulong(params.tile) - 1ul) /
                   ulong(params.tile))
            : 0u;
  uint outer_state = seed[params.seed_offset_words];
  if (!alive) {
    fail_seed(control, params.count_overflow_reason,
              params.declared_step, 0u);
    control->overflow_ordinal = ulong(params.maximum);
  }

  for (uint outer = 0u; outer < active_outer && alive; ++outer) {
    const ulong base = ulong(outer) * ulong(params.tile);
    const uint live =
        uint(min(ulong(params.tile), ulong(items) - base));
    const uint low =
        tile_low[params.tile_low_offset_words + ulong(outer)];
    const uint status =
        tile_status[params.tile_status_offset_words + ulong(outer)];
    if (status < params.tile) {
      // The first bounded Gather (queue slice) is proven in range. The
      // second Gather reports the invalid domain index, so canonical
      // telemetry includes exactly the first successful operation.
      StepControl telemetry =
          bounded_telemetry(live, params.tile, 1u);
      telemetry.overflow_ordinal = ulong(status);
      store_bounded_telemetry(
          control,
          bounded_prefix_telemetry(outer, live, params.tile, 1u));
      if (params.profile_steps != 0u) {
        steps[params.profile_seed_first + outer] = telemetry;
      }
      fail_seed(control, params.gather_reason,
                params.declared_step + outer, outer);
      // Canonical Gather telemetry reports the failing ordinal within the
      // current bounded invocation. The outer coordinate is recorded
      // separately, preserving the full deterministic failure identity.
      control->overflow_ordinal = ulong(status);
      alive = false;
    } else if (status == params.tile) {
      const StepControl telemetry =
          bounded_telemetry(live, params.tile, 2u);
      store_bounded_telemetry(
          control,
          bounded_prefix_telemetry(outer, live, params.tile, 2u));
      if (params.profile_steps != 0u) {
        steps[params.profile_seed_first + outer] = telemetry;
      }
      fail_seed(control, params.reduce_reason,
                params.declared_step + outer, outer);
      alive = false;
    } else {
      const StepControl telemetry =
          bounded_telemetry(live, params.tile, 2u);
      if (params.profile_steps != 0u) {
        steps[params.profile_seed_first + outer] = telemetry;
      }
      // Common admission proves Action is `tile_state + invariant` in U32.
      // Repeating it N times is exactly addition in Z/(2^32):
      // state + N*invariant modulo 2^32. This removes the serial inner loop
      // without reassociation of Gather/Reduce or Fold failure order.
      const uint invariant = scalar_value(
          params.action.rhs, low, live, outer_state,
          params.action.immediate);
      const uint tile_state = low + params.inner_bound * invariant;
      outer_state = evaluate_add(params.fold, tile_state, live,
                                 outer_state);
      control->iteration_count = saturated_add(
          control->iteration_count, 1ul);
      control->executed_outer_window_count = saturated_add(
          control->executed_outer_window_count, 1ul);
      control->executed_inner_iteration_count = saturated_add(
          control->executed_inner_iteration_count,
          ulong(params.inner_bound));
    }
  }

  control->generation += params.generation_stride;
  if (control->reason == 0u) {
    const uint skipped = params.outer_bound - active_outer;
    control->generated_item_count = 2ul * ulong(items);
    control->generated_capacity =
        2ul * ulong(active_outer) * ulong(params.tile);
    control->indirect_dispatch_count = 2ul * ulong(active_outer);
    control->indirect_work_item_count = 2ul * ulong(items);
    control->skipped_iteration_count = ulong(skipped);
    control->skipped_outer_window_count = ulong(skipped);
    // Both factors are U32 and common admission proves active_outer is no
    // greater than outer_bound. Promotion happens before multiplication, so
    // the exact skipped-inner count is strictly smaller than 2^64. Deferred
    // publication is intentional: a failure stops before inactive tails and
    // canonical execution therefore does not count those tails as skipped.
    control->skipped_inner_iteration_count =
        ulong(skipped) * ulong(params.inner_bound);
    target[params.target_offset_words] = outer_state;
    control->failed_step = 0xffffffffu;
    control->verified_prefix = params.declared_step_count;
  } else {
    control->verified_prefix = control->failed_step;
  }
}
)rundmetal";

template <typename Sink>
[[nodiscard]] bool EmitMetalNestedAggregateSource(Sink &sink) noexcept(
    noexcept(sink.append(std::string_view{}))) {
  return sink.append(MetalNestedAggregatePreamble) &&
         EmitPipelineNestedPhaseContract(
             sink, PipelineNestedPhaseSourceLanguage::Metal) &&
         sink.append(MetalNestedAggregateBody);
}

} // namespace

std::string_view MetalNestedAggregateSource() noexcept {
  static const auto source = backend_source_recipe::materialize_fixed<
      MetalNestedAggregatePreamble.size() + MetalNestedAggregateBody.size() +
      1024u>(
      [](auto &sink) noexcept { return EmitMetalNestedAggregateSource(sink); });
  return source.text();
}

} // namespace rund::node::accel::detail

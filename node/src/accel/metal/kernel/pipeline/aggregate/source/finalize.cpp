#include "finalize.hpp"

namespace rund::node::accel::detail {
namespace {

inline constexpr std::string_view MetalNestedAggregateFinalize = R"rundmetal(
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

} // namespace

std::string_view MetalNestedAggregateFinalizeSource() noexcept {
  return MetalNestedAggregateFinalize;
}

} // namespace rund::node::accel::detail

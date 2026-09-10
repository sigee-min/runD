#include "common.hpp"

namespace rund::node::accel::detail {
namespace {

inline constexpr std::string_view MetalNestedAggregatePreamble = R"rundmetal(
#include <metal_stdlib>
using namespace metal;
)rundmetal";

inline constexpr std::string_view MetalNestedAggregateCommon = R"rundmetal(
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
)rundmetal";

} // namespace

std::string_view MetalNestedAggregatePreambleSource() noexcept {
  return MetalNestedAggregatePreamble;
}

std::string_view MetalNestedAggregateCommonSource() noexcept {
  return MetalNestedAggregateCommon;
}

} // namespace rund::node::accel::detail

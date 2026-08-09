#include "source.hpp"

namespace rund::node::accel::detail::metal_pipeline_status_source {

std::string_view preamble() noexcept {
  return R"rundmetal(
#include <metal_stdlib>
using namespace metal;
)rundmetal";
}

std::string_view abi() noexcept {
  return R"rundmetal(
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

)rundmetal";
}

} // namespace rund::node::accel::detail::metal_pipeline_status_source

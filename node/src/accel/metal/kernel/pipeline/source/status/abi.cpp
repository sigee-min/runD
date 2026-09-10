#include "source.hpp"

#include "../../abi/source/begin.def"

namespace {

// Keep PipelineControl, ResidentState, and StepControl as raw production
// source.  Only the seven selected parameter records are projected from the
// shared schemas below.
inline constexpr char MetalPipelineAbi[] =
    "\n"
#include "../../abi/schema/source.def"
#include "../../abi/schema/entry.def"
#include "../../abi/schema/reset.def"
#include "../../abi/schema/status.def"
    R"rundmetal(struct PipelineControl {
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

)rundmetal"
#include "../../abi/schema/publish.def"
#include "../../abi/schema/window.def"
    R"rundmetal(struct ResidentState {
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

} // namespace

#include "../../abi/source/end.def"

namespace rund::node::accel::detail::metal_pipeline_status_source {

std::string_view preamble() noexcept {
  return R"rundmetal(
#include <metal_stdlib>
using namespace metal;
)rundmetal";
}

std::string_view abi() noexcept {
  return std::string_view{MetalPipelineAbi, sizeof(MetalPipelineAbi) - 1u};
}

} // namespace rund::node::accel::detail::metal_pipeline_status_source

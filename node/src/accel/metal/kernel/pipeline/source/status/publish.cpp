#include "source.hpp"

namespace rund::node::accel::detail::metal_pipeline_status_source {

std::string_view publish() noexcept {
  return R"rundmetal(kernel void rund_pipeline_publish(
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

)rundmetal";
}

} // namespace rund::node::accel::detail::metal_pipeline_status_source

#include "local.hpp"

namespace rund::node::accel::detail::device_vsm_window_projection {

bool exact_window_pipeline(const prepared::PipelineState &pipeline,
                           WindowAuthority &first, const char *&reason) {
  reason = "device_vsm_window_pipeline_invalid";
  if (pipeline.state_count == 0u || pipeline.states == nullptr ||
      pipeline.size == 0u) {
    return false;
  }
  if (!window_detail::exact_step(pipeline.states[0u].get(), first, reason)) {
    return false;
  }
  for (std::size_t index = 1u; index < pipeline.state_count; ++index) {
    WindowAuthority candidate{};
    if (!window_detail::exact_step(pipeline.states[index].get(), candidate,
                                   reason)) {
      return false;
    }
    if (!window_detail::same_authority(first, candidate)) {
      reason = "device_vsm_window_state_mismatch";
      return false;
    }
  }
  reason = "ok";
  return true;
}

bool same_window_pipeline(const prepared::PipelineState &pipeline,
                          const WindowAuthority &first) {
  WindowAuthority candidate{};
  const char *reason = nullptr;
  return exact_window_pipeline(pipeline, candidate, reason) &&
         window_detail::same_authority(first, candidate);
}

} // namespace rund::node::accel::detail::device_vsm_window_projection

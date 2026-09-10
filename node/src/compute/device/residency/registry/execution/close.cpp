#include "../../registry.hpp"

namespace rund::compute::detail::residency {

bool Authority::close_rows_clear_locked() const noexcept {
  const auto clear = [this](const std::uint32_t frame) noexcept {
    return frame < frames_.size() && frames_[frame].view_commit_stamp == 0u;
  };
  if (execution_state_.slot.frame_count > execution_state_.slot.frames.size() ||
      execution_state_.slot.undo_count >
          execution_state_.slot.undo_frames.size() ||
      execution_state_.slot.service_binding_count[0u] >
          execution_state_.slot.service_bindings[0u].size() ||
      execution_state_.slot.service_binding_count[1u] >
          execution_state_.slot.service_bindings[1u].size()) {
    return false;
  }
  for (std::size_t index = 0u; index < execution_state_.slot.frame_count;
       ++index) {
    if (!clear(execution_state_.slot.frames[index])) {
      return false;
    }
  }
  for (std::size_t index = 0u; index < execution_state_.slot.undo_count;
       ++index) {
    if (!clear(execution_state_.slot.undo_frames[index])) {
      return false;
    }
  }
  for (std::size_t service = 0u;
       service < execution_state_.slot.service_bindings.size(); ++service) {
    for (std::size_t index = 0u;
         index < execution_state_.slot.service_binding_count[service];
         ++index) {
      if (!clear(
              execution_state_.slot.service_bindings[service][index].frame)) {
        return false;
      }
    }
  }
  for (std::size_t phase = 0u;
       phase < execution_state_.slot.window_service_bindings.size(); ++phase) {
    for (std::size_t bank = 0u;
         bank < execution_state_.slot.window_service_bindings[phase].size();
         ++bank) {
      if (execution_state_.slot.window_service_binding_count[phase][bank] >
          execution_state_.slot.window_service_bindings[phase][bank].size()) {
        return false;
      }
      for (std::size_t index = 0u;
           index <
           execution_state_.slot.window_service_binding_count[phase][bank];
           ++index) {
        if (!clear(execution_state_.slot
                       .window_service_bindings[phase][bank][index]
                       .frame)) {
          return false;
        }
      }
    }
  }
  return true;
}

} // namespace rund::compute::detail::residency

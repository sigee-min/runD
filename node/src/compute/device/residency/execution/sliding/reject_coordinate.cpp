#include "internal.hpp"

namespace rund::compute::detail::residency {
namespace {

[[nodiscard]] registry_model::Frame
empty_service_frame(const registry_model::Frame &frame) noexcept {
  return registry_model::Frame{
      .state = FrameState::Empty,
      .tier = frame.tier,
      .role = frame.role,
      .extent = frame.extent,
      .view = frame.view,
      .assigned = frame.assigned,
  };
}

} // namespace

bool SlidingOwner::reject_execution_sliding_coordinate(
    const execution::Plan &plan, execution::Sliding &sliding,
    const std::uint64_t coordinate, const Status failure) noexcept {
  if (sliding.state_ == nullptr || failure) {
    return false;
  }
  std::lock_guard authority_lock{authority_.gate_};
  if (authority_.view_commit_quarantined_locked()) {
    return false;
  }
  execution::Sliding::State &state = *sliding.state_;
  std::lock_guard state_lock{state.gate};
  if (!authority_.execution_state_.slot.sliding_admitted ||
      authority_.execution_state_.slot.window_final ||
      authority_.execution_state_.slot.token != state.token ||
      authority_.execution_state_.slot.generation != state.generation ||
      authority_.execution_state_.slot.native_inflight != state.owner ||
      !state.authority_bound || state.model_only || state.closed ||
      state.finalizing || coordinate != state.admitted ||
      coordinate >= state.planned ||
      state.terminal_frontier != state.admitted ||
      state.persist_frontier != state.persist_issued || plan.identity() == 0u ||
      plan.identity() != authority_.execution_state_.slot.plan ||
      state.invocation.direct_ == nullptr ||
      state.invocation.direct_->identity() != plan.identity()) {
    return false;
  }
  if (state.has_failure) {
    if (state.first_failure.ordinal != coordinate || state.status ||
        state.status.reason() != failure.reason() ||
        state.first_failure_terminal != execution::TerminalKind::Known ||
        state.terminal != execution::TerminalKind::Known || state.quarantine) {
      return false;
    }
    for (std::size_t index = 0u; index < state.input_count; ++index) {
      const execution::Sliding::State::InputCell &cell = state.inputs[index];
      if (cell.state == execution::InputState::Free) {
        continue;
      }
      if (cell.state != execution::InputState::Invalidating ||
          cell.coordinate.ordinal != coordinate || cell.physical_handoff ||
          cell.physical_frame >= authority_.frames_.size() ||
          !authority_.frames_[cell.physical_frame].assigned ||
          authority_.frames_[cell.physical_frame].state !=
              FrameState::Mapping) {
        return false;
      }
    }
    if (std::any_of(state.outputs.begin(),
                    state.outputs.begin() + state.output_count,
                    [](const auto &cell) {
                      return cell.state != execution::OutputState::Free;
                    }) ||
        std::any_of(state.native_cells.begin(), state.native_cells.end(),
                    [](const auto &cell) {
                      return cell.state != execution::NativeState::Free;
                    }) ||
        std::any_of(state.terminals.begin(), state.terminals.end(),
                    [](const auto &cell) { return cell.invalidating; })) {
      return false;
    }
    for (std::size_t index = 0u; index < state.input_count; ++index) {
      execution::Sliding::State::InputCell &cell = state.inputs[index];
      if (cell.state != execution::InputState::Invalidating) {
        continue;
      }
      authority_.frames_[cell.physical_frame] =
          empty_service_frame(authority_.frames_[cell.physical_frame]);
      cell.state = execution::InputState::Free;
    }
    return state.no_issued();
  }
  if (!state.no_issued()) {
    return false;
  }
  std::array<PageUse, execution::UseCapacity> uses{};
  execution::SlidingProjection projection{};
  if (!state.invocation.project(coordinate, uses, projection)) {
    return false;
  }
  if (projection.coordinate.ordinal != coordinate) {
    return false;
  }
  state.fail(projection.coordinate, failure, execution::TerminalKind::Known,
             false);
  return true;
}

} // namespace rund::compute::detail::residency

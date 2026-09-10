#include "../../../registry/execution_owner.hpp"

#include "../../../execution/plan.hpp"
#include "../../frame.hpp"

#include <algorithm>
#include <mutex>

namespace rund::compute::detail::residency {

bool ExecutionOwner::abandon_execution(const std::uint64_t token,
                                       const std::uint64_t generation,
                                       const execution::Plan &plan) noexcept {
  std::lock_guard lock{authority_.gate_};
  if (authority_.view_commit_quarantined_locked() || token == 0u ||
      generation == 0u || authority_.execution_state_.slot.token != token ||
      authority_.execution_state_.slot.sliding_admitted ||
      authority_.execution_state_.slot.generation != generation ||
      authority_.execution_state_.slot.plan != plan.identity() ||
      authority_.execution_state_.slot.epochs != plan.epoch_count() ||
      (!authority_.execution_state_.slot.cache_admitted &&
       !authority_.execution_state_.slot.window_cache_admitted) ||
      authority_.execution_state_.slot.release_count != 0u ||
      authority_.execution_state_.slot.window_final ||
      authority_.execution_state_.slot.native_rejected ||
      authority_.execution_state_.slot.native_inflight != 0u ||
      (authority_.execution_state_.slot.window_cache_admitted &&
       (authority_.execution_state_.slot.native_accepted ||
        authority_.execution_state_.slot.progress.native_dispatches != 0u ||
        authority_.execution_state_.slot.progress.native_completions != 0u ||
        authority_.execution_state_.slot.progress.output_services != 0u))) {
    return false;
  }
  if (authority_.execution_state_.slot.window_cache_admitted) {
    for (std::size_t index = 0u;
         index < authority_.execution_state_.slot.undo_count; ++index) {
      authority_.frames_[authority_.execution_state_.slot.undo_frames[index]] =
          authority_.execution_state_.slot.undo[index];
    }
  } else {
    for (std::size_t index = 0u;
         index < authority_.execution_state_.slot.frame_count; ++index) {
      const std::uint32_t frame =
          authority_.execution_state_.slot.frames[index];
      authority_.frames_[frame] =
          frame_detail::empty(authority_.frames_[frame]);
    }
  }
  authority_.execution_state_.slot = registry_model::ExecutionSlot{};
  return true;
}

bool ExecutionOwner::abandon_execution_window(
    const execution::Plan &plan,
    const execution::WindowEvidence &evidence) noexcept {
  std::lock_guard lock{authority_.gate_};
  if (authority_.view_commit_quarantined_locked() ||
      authority_.execution_state_.slot.token == 0u ||
      !authority_.execution_state_.slot.window_cache_admitted ||
      authority_.execution_state_.slot.sliding_admitted ||
      authority_.execution_state_.slot.window_final ||
      authority_.execution_state_.slot.native_accepted ||
      authority_.execution_state_.slot.native_rejected ||
      authority_.execution_state_.slot.epochs == 0u ||
      authority_.execution_state_.slot.epochs > execution::WindowCapacity ||
      authority_.execution_state_.slot.plan != plan.identity() ||
      authority_.execution_state_.slot.epochs != plan.epoch_count() ||
      evidence.plan_identity != authority_.execution_state_.slot.plan ||
      evidence.token != authority_.execution_state_.slot.token ||
      evidence.generation != authority_.execution_state_.slot.generation ||
      evidence.epoch_count != authority_.execution_state_.slot.epochs ||
      evidence.public_handoffs != 1u ||
      evidence.terminal != execution::TerminalKind::Known ||
      evidence.release_count != authority_.execution_state_.slot.epochs ||
      evidence.release_count > evidence.releases.size() ||
      evidence.native_batches > evidence.release_count ||
      evidence.queue_calls > evidence.native_batches) {
    return false;
  }
  std::uint64_t dispatched = 0u;
  for (std::size_t index = 0u; index < evidence.release_count; ++index) {
    const execution::Release &release = evidence.releases[index];
    if (release.plan_identity != authority_.execution_state_.slot.plan ||
        release.token != authority_.execution_state_.slot.token ||
        release.generation != authority_.execution_state_.slot.generation ||
        release.epoch != index || release.backend_sequence != index + 1u ||
        release.bank != index % execution::BankCapacity ||
        release.terminal != execution::TerminalKind::Known ||
        !release.dispatched || !release.completed) {
      return false;
    }
    ++dispatched;
  }
  if (dispatched != evidence.native_batches) {
    return false;
  }
  for (std::size_t index = 0u;
       index < authority_.execution_state_.slot.frame_count; ++index) {
    const std::uint32_t frame = authority_.execution_state_.slot.frames[index];
    authority_.frames_[frame] = frame_detail::empty(authority_.frames_[frame]);
  }
  authority_.execution_state_.slot = registry_model::ExecutionSlot{};
  return true;
}

bool ExecutionOwner::abandon_execution_stream(
    const execution::Plan &plan,
    const execution::WindowEvidence &evidence) noexcept {
  std::lock_guard lock{authority_.gate_};
  if (authority_.view_commit_quarantined_locked() ||
      authority_.execution_state_.slot.token == 0u ||
      !authority_.execution_state_.slot.window_cache_admitted ||
      authority_.execution_state_.slot.sliding_admitted ||
      authority_.execution_state_.slot.window_final ||
      authority_.execution_state_.slot.native_accepted ||
      authority_.execution_state_.slot.native_rejected ||
      authority_.execution_state_.slot.epochs == 0u ||
      authority_.execution_state_.slot.plan != plan.identity() ||
      authority_.execution_state_.slot.epochs != plan.epoch_count() ||
      evidence.plan_identity != authority_.execution_state_.slot.plan ||
      evidence.token != authority_.execution_state_.slot.token ||
      evidence.generation != authority_.execution_state_.slot.generation ||
      evidence.epoch_count != authority_.execution_state_.slot.epochs ||
      evidence.public_handoffs != 1u ||
      evidence.terminal != execution::TerminalKind::Known ||
      evidence.first_epoch !=
          authority_.execution_state_.slot.window_accept_count ||
      evidence.release_count == 0u ||
      evidence.release_count > execution::WindowCapacity ||
      evidence.release_count !=
          std::min<std::uint64_t>(execution::WindowCapacity,
                                  authority_.execution_state_.slot.epochs -
                                      evidence.first_epoch) ||
      authority_.execution_state_.slot.release_count !=
          evidence.first_epoch + evidence.release_count) {
    return false;
  }
  for (std::size_t index = 0u; index < evidence.release_count; ++index) {
    const std::uint64_t epoch = evidence.first_epoch + index;
    const std::size_t slot =
        static_cast<std::size_t>(epoch % execution::WindowCapacity);
    const execution::Release &release = evidence.releases[index];
    const execution::Release &journal =
        authority_.execution_state_.slot.releases[slot];
    if (authority_.execution_state_.slot.release_epochs[slot] != epoch ||
        release.plan_identity != authority_.execution_state_.slot.plan ||
        release.token != authority_.execution_state_.slot.token ||
        release.generation != authority_.execution_state_.slot.generation ||
        release.epoch != epoch || release.backend_sequence != epoch + 1u ||
        release.bank != epoch % execution::BankCapacity ||
        release.terminal != execution::TerminalKind::Known ||
        !release.dispatched || !release.completed ||
        release.status.reason() != journal.status.reason() ||
        release.may_write != journal.may_write ||
        release.completed != journal.completed ||
        release.dispatched != journal.dispatched) {
      return false;
    }
  }
  for (std::size_t index = 0u;
       index < authority_.execution_state_.slot.undo_count; ++index) {
    authority_.frames_[authority_.execution_state_.slot.undo_frames[index]] =
        authority_.execution_state_.slot.undo[index];
  }
  for (std::size_t index = 0u;
       index < authority_.execution_state_.slot.frame_count; ++index) {
    const std::uint32_t frame = authority_.execution_state_.slot.frames[index];
    authority_.frames_[frame] = frame_detail::empty(authority_.frames_[frame]);
  }
  authority_.execution_state_.slot = registry_model::ExecutionSlot{};
  return true;
}

} // namespace rund::compute::detail::residency

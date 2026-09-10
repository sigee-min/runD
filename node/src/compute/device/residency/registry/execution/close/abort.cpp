#include "../../../registry/execution_owner.hpp"

#include "../../../execution/plan.hpp"
#include "../../frame.hpp"

#include <mutex>

namespace rund::compute::detail::residency {

ExecutionClose ExecutionOwner::abort_execution_stream(
    const std::uint64_t token, const std::uint64_t generation,
    const execution::Plan &plan, const Status failure) noexcept {
  std::lock_guard lock{authority_.gate_};
  if (authority_.view_commit_quarantined_locked() || failure || token == 0u ||
      generation == 0u || authority_.execution_state_.slot.token != token ||
      authority_.execution_state_.slot.sliding_admitted ||
      authority_.execution_state_.slot.generation != generation ||
      authority_.execution_state_.slot.plan != plan.identity() ||
      authority_.execution_state_.slot.epochs != plan.epoch_count() ||
      !authority_.execution_state_.slot.window_cache_admitted ||
      authority_.execution_state_.slot.window_final ||
      authority_.execution_state_.slot.native_accepted ||
      authority_.execution_state_.slot.native_rejected ||
      authority_.execution_state_.slot.release_count == 0u ||
      authority_.execution_state_.slot.release_count !=
          authority_.execution_state_.slot.window_accept_count ||
      authority_.execution_state_.slot.release_count >=
          authority_.execution_state_.slot.epochs ||
      authority_.execution_state_.slot.native_inflight != 0u ||
      authority_.execution_state_.slot.failed ||
      authority_.execution_state_.slot.unknown ||
      authority_.execution_state_.slot.progress.input_services !=
          authority_.execution_state_.slot.release_count ||
      authority_.execution_state_.slot.progress.output_services !=
          authority_.execution_state_.slot.release_count ||
      authority_.execution_state_.slot.progress.native_dispatches !=
          authority_.execution_state_.slot.release_count ||
      authority_.execution_state_.slot.progress.native_completions !=
          authority_.execution_state_.slot.release_count) {
    return ExecutionClose{.failure = AuthorityFailure::Invalid};
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
  const ExecutionClose result{
      .failure = AuthorityFailure::None,
      .progress = authority_.execution_state_.slot.progress,
      .success = false,
      .quarantined = false,
  };
  authority_.execution_state_.slot = registry_model::ExecutionSlot{};
  return result;
}

} // namespace rund::compute::detail::residency

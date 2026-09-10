#include "internal.hpp"

namespace rund::node::accel::detail::prepared::sliding {

void complete_native(void *const raw, KernelResult result) noexcept {
  auto *const slot = static_cast<Slot *>(raw);
  if (slot == nullptr || slot->state == nullptr) {
    return;
  }
  State &state = *slot->state;
  const std::shared_ptr<State> callback_owner = state.shared_from_this();
  (void)callback_owner;
  {
    std::lock_guard lock{state.gate};
    if (slot->phase == SlotPhase::Submitting) {
      slot->inline_result = std::move(result);
      slot->inline_terminal = true;
      slot->same_thread_inline = SubmittingSlot == slot;
      return;
    }
  }
  slot->submit_returned.wait(false, std::memory_order_acquire);

  PreparedResidencySlidingReleaseCompletion completion = nullptr;
  void *user = nullptr;
  PreparedResidencySlidingRelease release{};
  std::shared_ptr<State> retained{};
  {
    std::lock_guard lock{state.gate};
    if (!state.active || state.final_sent ||
        slot->phase != SlotPhase::Submitted) {
      return;
    }
    retained = state.shared_from_this();
    prepared::PipelineState *const pipeline = state.pipelines[slot->slot];
    PreparedPipelineEvidence evidence = prepared::PipelineEvidence(
        state.context, *pipeline, result,
        std::span<const std::uint32_t>{slot->selection.locals.data(),
                                       slot->selection.local_count});
    const std::uint64_t native_calls =
        evidence.shared.run.work.command_submit_count;
    const bool known = result.terminal == NativeTerminal::Known;
    const bool malformed =
        native_calls != 1u || !evidence.submitted ||
        evidence.terminal != result.terminal ||
        evidence.check.ok != result.check.ok ||
        evidence.control.generation != slot->selection.control_generation;
    if (state.inflight != 0u) {
      --state.inflight;
    }
    state.queue_calls += native_calls;
    release = PreparedResidencySlidingRelease{
        .terminal =
            BackendResidencySlidingTerminal{
                .check = malformed
                             ? rund::AccelCheck{false,
                                                "accel_kernel_pipeline_invalid"}
                             : evidence.check,
                .terminal = malformed ? NativeTerminal::UnknownMayWrite
                                      : result.terminal,
                .coordinate = slot->coordinate,
                .turn = slot->turn,
                .queue_calls = native_calls,
                .slot = slot->slot,
                .dispatched = true,
                .completed = known && !malformed,
                .may_write = slot->selection.write_mask != 0u,
            },
        .evidence = std::move(evidence),
    };
    completion = state.request.release;
    user = state.request.user;
    ++state.external_calls;
    slot->phase = SlotPhase::Returning;
    if (malformed || !known) {
      state.unknown = true;
      record_failure(state, slot->coordinate, {false, "compute_device_lost"});
    } else if (!release.terminal.check.ok) {
      record_failure(state, slot->coordinate, release.terminal.check);
    }
  }
  completion(user, std::move(release));
  bool resume_service = true;
  {
    std::lock_guard lock{state.gate};
    if (state.external_calls != 0u) {
      --state.external_calls;
    }
    resume_service = state.active && !state.final_sent;
  }
  if (resume_service) {
    schedule_service(*retained);
  }
  emit_final(retained);
}

} // namespace rund::node::accel::detail::prepared::sliding

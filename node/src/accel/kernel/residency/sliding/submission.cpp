#include "internal.hpp"

namespace rund::node::accel::detail::prepared::sliding {

thread_local Slot *SubmittingSlot = nullptr;

namespace {

[[nodiscard]] rund::AccelCheck submit_native(State &state, Slot &slot,
                                             PipelineState &pipeline,
                                             const bool allowed) noexcept {
  if (!allowed) {
    return {false, "accel_kernel_pipeline_invalid"};
  }
  Slot *const prior_submitting = SubmittingSlot;
  SubmittingSlot = &slot;
  const rund::AccelCheck submitted = pipeline.ops->submit_prepared_sliding(
      pipeline.backend,
      BackendResidencySlidingDescriptor{
          .owner = pipeline.backend.get(),
          .plan_identity = state.request.plan_identity,
          .token = state.request.token,
          .generation = state.request.generation,
          .coordinate = slot.coordinate,
          .turn = slot.turn,
          .read_mask = slot.selection.read_mask,
          .write_mask = slot.selection.write_mask,
          .descriptor_generation = slot.selection.descriptor_generation,
          .control_generation = slot.selection.control_generation,
          .stride = static_cast<std::uint8_t>(state.role_count),
          .slot = slot.slot,
      },
      complete_native, &slot, KernelTiming::Submission,
      PipelineSubmitMode::Residency,
      std::span<const std::uint32_t>{slot.selection.locals.data(),
                                     slot.selection.local_count});
  SubmittingSlot = prior_submitting;
  return submitted;
}

void publish_submit_return(State &state, Slot &slot, const bool native_submit,
                           const rund::AccelCheck submitted) noexcept {
  bool inline_terminal = false;
  bool inline_unknown = false;
  KernelResult inline_result{};
  bool final = false;
  {
    std::lock_guard lock{state.gate};
    if (!state.active || slot.phase != SlotPhase::Submitting) {
      slot.submit_returned.store(true, std::memory_order_release);
      slot.submit_returned.notify_all();
      return;
    }
    if (slot.inline_terminal) {
      slot.phase = SlotPhase::Submitted;
      ++state.accepted;
      ++state.inflight;
      state.inflight_peak = std::max(state.inflight_peak, state.inflight);
      inline_terminal = true;
      inline_unknown = slot.same_thread_inline || !submitted.ok;
      inline_result = std::move(slot.inline_result);
    } else if (!native_submit && state.failed) {
      slot.phase = SlotPhase::Done;
      final = state.inflight == 0u;
    } else if (!submitted.ok) {
      if (state.failed) {
        slot.phase = SlotPhase::Done;
      } else {
        fail_projection(state, slot, submitted);
      }
      final = state.inflight == 0u;
    } else {
      slot.phase = SlotPhase::Submitted;
      ++state.accepted;
      ++state.inflight;
      state.inflight_peak = std::max(state.inflight_peak, state.inflight);
      inline_terminal = slot.inline_terminal;
      inline_result = std::move(slot.inline_result);
    }
    slot.submit_returned.store(true, std::memory_order_release);
    slot.submit_returned.notify_all();
  }
  if (inline_terminal) {
    if (inline_unknown) {
      inline_result.check = {false, "compute_device_lost"};
      inline_result.terminal = NativeTerminal::UnknownMayWrite;
    }
    complete_native(&slot, std::move(inline_result));
  } else if (final) {
    emit_final(state.shared_from_this());
  }
}

} // namespace

void submit_slot(const std::shared_ptr<State> &state,
                 const std::size_t index) noexcept {
  Slot &slot = state->slots[index];
  PipelineState &pipeline = *state->pipelines[index];
  {
    std::lock_guard lock{state->gate};
    ++state->external_calls;
  }
  const rund::AccelCheck seeded =
      pipeline.ops->seed_prepared_pipeline_generation(
          pipeline.backend, slot.selection.control_generation - 1u);
  bool native_submit = false;
  {
    std::lock_guard lock{state->gate};
    native_submit = seeded.ok && state->active && !state->failed &&
                    slot.phase == SlotPhase::Submitting;
  }
  const rund::AccelCheck submitted =
      native_submit ? submit_native(*state, slot, pipeline, true) : seeded;
  {
    std::lock_guard lock{state->gate};
    if (state->external_calls != 0u) {
      --state->external_calls;
    }
  }
  publish_submit_return(*state, slot, native_submit, submitted);
}

} // namespace rund::node::accel::detail::prepared::sliding

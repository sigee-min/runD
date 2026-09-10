#include "internal.hpp"

#include <algorithm>
#include <array>
#include <limits>
#include <memory>
#include <mutex>
#include <utility>

namespace rund::node::accel::detail::prepared_residency {

void CompleteResidencyWindowRelease(
    void *const raw, BackendResidencyWindowRelease &&release) noexcept {
  auto *const control = static_cast<PreparedResidencyWindowControl *>(raw);
  if (control == nullptr) {
    return;
  }
  PreparedResidencyWindowReleaseCompletion completion = nullptr;
  void *user = nullptr;
  PreparedKernelPipeline pipeline{};
  std::array<std::uint32_t, ResidencyWindowLocalCapacity> locals{};
  std::size_t local_count = 0u;
  bool valid = false;
  {
    std::lock_guard lock{control->gate};
    if (!control->active || control->quarantined ||
        control->release_count >= control->bound.batch_count ||
        control->release_count >= ResidencyWindowCapacity ||
        control->bound.first_epoch >
            std::numeric_limits<std::uint64_t>::max() - control->release_count ||
        release.receipt.epoch !=
            control->bound.first_epoch + control->release_count) {
      return;
    }
    const std::size_t index = control->release_count;
    const PreparedResidencyWindowBatch &batch = control->bound.batches[index];
    valid = release.receipt.bank == batch.bank &&
            release.receipt.backend_sequence == batch.epoch + 1u &&
            release.receipt.dispatched && release.result.pipeline.submitted &&
            SameCheck(release.receipt.check, release.result.check) &&
            release.receipt.terminal == release.result.terminal &&
            release.receipt.completed ==
                (release.receipt.terminal == NativeTerminal::Known) &&
            (release.receipt.terminal != NativeTerminal::UnknownMayWrite ||
             release.receipt.may_write) &&
            (!release.receipt.check.ok ||
             (release.receipt.completed && release.receipt.may_write));
    if (!valid) {
      release.receipt.check = {false, "accel_kernel_pipeline_invalid"};
      if (release.receipt.terminal == NativeTerminal::UnknownMayWrite ||
          release.result.terminal == NativeTerminal::UnknownMayWrite) {
        release.receipt.terminal = NativeTerminal::UnknownMayWrite;
        release.receipt.completed = false;
        release.receipt.may_write = true;
      }
    }
    control->receipts[index] = release.receipt;
    pipeline = batch.pipeline;
    locals = batch.locals;
    local_count = batch.local_count;
    completion = control->bound.release;
    user = control->bound.user;
    ++control->release_count;
  }
  auto *const state =
      static_cast<prepared::PipelineState *>(pipeline.owner.get());
  PreparedPipelineEvidence evidence{};
  if (valid && state != nullptr) {
    evidence = prepared::PipelineEvidence(
        state->context, *state, release.result,
        std::span<const std::uint32_t>{locals.data(), local_count});
  } else {
    evidence.check = release.receipt.check;
  }
  completion(user, PreparedResidencyWindowRelease{
                       .receipt = release.receipt,
                       .evidence = std::move(evidence),
                   });
}

void CompleteResidencyWindowFinal(
    void *const raw, BackendResidencyWindowFinal &&final) noexcept {
  auto *const control = static_cast<PreparedResidencyWindowControl *>(raw);
  if (control == nullptr) {
    return;
  }
  PreparedResidencyWindowFinalCompletion completion = nullptr;
  void *user = nullptr;
  std::array<prepared::PipelineState *, ResidencyWindowCapacity> states{};
  // Keep every prepared owner alive across claim release, control clearing,
  // and the outer final callback. Otherwise clearing bound/submission.owner
  // can destroy a PipelineState while its own mutex is still locked.
  std::array<PreparedKernelPipeline, ResidencyWindowCapacity> owners{};
  std::size_t state_count = 0u;
  bool quarantine = false;
  {
    std::unique_lock lock{control->gate};
    if (!control->active || control->quarantined) {
      return;
    }
    bool saw_unknown = false;
    const BackendResidencyWindowReceipt *first_failure = nullptr;
    bool valid = final.plan_identity == control->bound.plan_identity &&
                 final.token == control->bound.token &&
                 final.generation == control->bound.generation &&
                 final.first_epoch == control->bound.first_epoch &&
                 final.public_handoffs == 1u &&
                 final.native_batches == control->bound.batch_count &&
                 final.queue_calls != 0u &&
                 final.queue_calls <= final.native_batches &&
                 final.receipt_count == control->bound.batch_count &&
                 control->release_count == control->bound.batch_count;
    for (std::size_t index = 0u; index < control->release_count; ++index) {
      saw_unknown = saw_unknown || control->receipts[index].terminal ==
                                       NativeTerminal::UnknownMayWrite;
    }
    const std::size_t final_receipt_count =
        std::min(final.receipt_count, ResidencyWindowCapacity);
    for (std::size_t index = 0u; index < final_receipt_count; ++index) {
      const BackendResidencyWindowReceipt &receipt = final.receipts[index];
      const PreparedResidencyWindowBatch &batch = control->bound.batches[index];
      const bool row_valid =
          receipt.epoch == batch.epoch &&
          receipt.backend_sequence == batch.epoch + 1u &&
          receipt.bank == batch.bank && receipt.dispatched &&
          SameWindowReceipt(receipt, control->receipts[index]);
      valid = valid && row_valid;
      if (first_failure == nullptr && !receipt.check.ok) {
        first_failure = &receipt;
      }
    }
    valid =
        valid &&
        (saw_unknown == (final.terminal == NativeTerminal::UnknownMayWrite)) &&
        ((first_failure == nullptr && final.check.ok) ||
         (first_failure != nullptr &&
          SameCheck(final.check, first_failure->check))) &&
        final.completed_ns != 0u;
    if (!valid) {
      // A malformed/premature Final cannot prove that every may-write native
      // command reached a terminal. Collapsing it to a Known failure would
      // release prepared owners for reuse while a command may still execute.
      final.check = {false, "accel_kernel_pipeline_invalid"};
      final.terminal = NativeTerminal::UnknownMayWrite;
    }
    quarantine =
        !valid || saw_unknown ||
        final.terminal == NativeTerminal::UnknownMayWrite;
    completion = control->bound.final;
    user = control->bound.user;
    for (std::size_t index = 0u; index < control->bound.batch_count; ++index) {
      owners[index] = control->bound.batches[index].pipeline;
    }
    for (prepared::PipelineState *const state : control->states) {
      if (state == nullptr) {
        continue;
      }
      std::size_t position = 0u;
      while (position < state_count &&
             std::less<prepared::PipelineState *>{}(states[position], state)) {
        ++position;
      }
      if (position < state_count && states[position] == state) {
        continue;
      }
      // There are at most four owners. Insert directly into canonical mutex
      // order instead of instantiating a general-purpose introsort.
      for (std::size_t next = state_count; next > position; --next) {
        states[next] = states[next - 1u];
      }
      states[position] = state;
      ++state_count;
    }
    if (quarantine) {
      control->quarantined = true;
    }
    // Never hold control.gate while acquiring a prepared submission mutex.
    // Submit claims submission mutexes before control.gate; preserving that
    // global order avoids an AB/BA race with a concurrent resubmit. Unknown
    // converts the raw caller-owned claim into a submission-owned quarantine;
    // Known releases both the claim and its strong owner.
    lock.unlock();
    std::array<std::unique_lock<std::mutex>, ResidencyWindowCapacity> claims{};
    for (std::size_t index = 0u; index < state_count; ++index) {
      claims[index] =
          std::unique_lock<std::mutex>{states[index]->submission.mutex};
    }
    PreparedResidencyStreamControl *const stream = control->stream;
    std::unique_lock<std::mutex> stream_lock{};
    if (stream != nullptr) {
      stream_lock = std::unique_lock<std::mutex>{stream->gate};
    }
    lock.lock();
    if (!control->active || control->quarantined != quarantine) {
      return;
    }
    for (std::size_t index = 0u; index < state_count; ++index) {
      prepared::PipelineSubmission &submission = states[index]->submission;
      if (submission.window == control) {
        submission.window = nullptr;
      }
      submission.quarantined = quarantine;
      if (!quarantine && submission.stream == nullptr) {
        submission.owner.reset();
      }
    }
    if (stream != nullptr && quarantine) {
      stream->quarantined = true;
    }
    control->bound = {};
    control->backend = {};
    control->states.fill(nullptr);
    control->receipts = {};
    control->release_count = 0u;
    control->stream = nullptr;
    control->active = false;
    control->aborting = false;
    lock.unlock();
  }
  completion(user, PreparedResidencyWindowFinal{.evidence = std::move(final)});
  static_cast<void>(owners);
}

} // namespace rund::node::accel::detail::prepared_residency

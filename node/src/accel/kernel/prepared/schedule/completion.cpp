#include "local.hpp"

#include "../evidence.hpp"
#include "../model.hpp"

#include <array>
#include <cstring>
#include <mutex>

namespace rund::node::accel::detail {
namespace {

[[nodiscard]] bool same_check(const rund::AccelCheck left,
                              const rund::AccelCheck right) noexcept {
  return left.ok == right.ok && left.reason != nullptr &&
         right.reason != nullptr && std::strcmp(left.reason, right.reason) == 0;
}

} // namespace

void CompletePreparedScheduleRelease(void *const raw,
                      BackendResidencyScheduleRelease &&release) noexcept {
  auto *const control = static_cast<PreparedResidencyScheduleControl *>(raw);
  if (control == nullptr) {
    return;
  }
  PreparedResidencyScheduleReleaseCompletion completion = nullptr;
  void *user = nullptr;
  PreparedKernelPipeline pipeline{};
  std::array<std::uint32_t, ResidencyWindowLocalCapacity> locals{};
  std::size_t local_count = 0u;
  bool valid = false;
  {
    std::lock_guard lock{control->gate};
    if (!control->active || control->quarantined ||
        control->release_count >= control->bound.epoch_count ||
        release.receipt.epoch != control->release_count) {
      return;
    }
    const std::size_t role = static_cast<std::size_t>(
        release.receipt.epoch % control->bound.role_count);
    const PreparedResidencyScheduleRole &source = control->bound.roles[role];
    local_count = release.receipt.epoch + 1u == control->bound.epoch_count
                      ? control->bound.tail_local_count
                      : source.local_count;
    const bool known = release.receipt.terminal == NativeTerminal::Known;
    valid = release.receipt.backend_sequence == release.receipt.epoch + 1u &&
            release.receipt.bank == release.receipt.epoch % 2u &&
            release.receipt.dispatched && release.result.pipeline.submitted &&
            same_check(release.receipt.check, release.result.check) &&
            release.receipt.terminal == release.result.terminal &&
            release.receipt.completed == known &&
            (known || release.receipt.may_write) &&
            (!release.receipt.check.ok ||
             (release.receipt.completed && release.receipt.may_write));
    if (!valid) {
      control->malformed = true;
      release.receipt.check = {false, "accel_kernel_pipeline_invalid"};
      release.receipt.terminal = NativeTerminal::UnknownMayWrite;
      release.receipt.completed = false;
      release.receipt.may_write = true;
    }
    if (release.receipt.check.ok && control->first_failure.ok) {
      ++control->success_prefix;
    } else if (!release.receipt.check.ok && control->first_failure.ok) {
      control->first_failure = release.receipt.check;
    } else {
      const bool suppressed =
          !release.receipt.check.ok &&
          release.receipt.terminal == NativeTerminal::Known &&
          release.receipt.completed && !release.receipt.may_write;
      if (!suppressed ||
          (control->suppressed_count != 0u &&
           control->suppressed_first + control->suppressed_count !=
               release.receipt.epoch)) {
        control->malformed = true;
      } else {
        if (control->suppressed_count == 0u) {
          control->suppressed_first = release.receipt.epoch;
        }
        ++control->suppressed_count;
      }
    }
    control->quarantined =
        control->quarantined ||
        release.receipt.terminal == NativeTerminal::UnknownMayWrite;
    ++control->release_count;
    pipeline = source.pipeline;
    locals = source.locals;
    completion = control->bound.release;
    user = control->bound.user;
  }
  PreparedPipelineEvidence evidence{};
  prepared::PipelineState *const state = PreparedScheduleState(pipeline);
  if (valid && state != nullptr) {
    evidence = prepared::PipelineEvidence(
        state->context, *state, release.result,
        std::span<const std::uint32_t>{locals.data(), local_count});
  } else {
    evidence.check = release.receipt.check;
    evidence.terminal = release.receipt.terminal;
  }
  completion(user, PreparedResidencyScheduleRelease{
                       .receipt = release.receipt,
                       .evidence = std::move(evidence),
                   });
}

void CompletePreparedScheduleFinal(void *const raw,
                    BackendResidencyScheduleFinal &&final) noexcept {
  auto *const control = static_cast<PreparedResidencyScheduleControl *>(raw);
  if (control == nullptr) {
    return;
  }
  PreparedResidencyScheduleFinalCompletion completion = nullptr;
  void *user = nullptr;
  std::array<PreparedKernelPipeline, ResidencyScheduleRoleCapacity> owners{};
  std::array<prepared::PipelineState *, ResidencyScheduleRoleCapacity> states{};
  std::size_t state_count = 0u;
  bool quarantine = false;
  {
    std::unique_lock lock{control->gate};
    if (!control->active) {
      return;
    }
    const bool unknown = control->quarantined ||
                         final.terminal == NativeTerminal::UnknownMayWrite;
    const bool success =
        control->first_failure.ok && !control->malformed && !unknown;
    const bool release_extent =
        unknown ? (control->release_count != 0u &&
                   control->release_count <= control->bound.epoch_count)
                : control->release_count == control->bound.epoch_count;
    bool valid =
        final.plan_identity == control->bound.plan_identity &&
        final.token == control->bound.token &&
        final.generation == control->bound.generation &&
        final.epoch_count == control->bound.epoch_count &&
        final.public_handoffs == 1u &&
        final.native_batches == control->bound.epoch_count &&
        final.queue_calls != 0u && final.queue_calls <= final.native_batches &&
        final.native_inflight_peak != 0u && final.native_inflight_peak <= 2u &&
        final.completed_ns != 0u && release_extent &&
        final.released_prefix == control->release_count &&
        final.completed_prefix == control->success_prefix &&
        final.suppressed_first == control->suppressed_first &&
        final.suppressed_count == control->suppressed_count &&
        final.terminal == (unknown ? NativeTerminal::UnknownMayWrite
                                   : NativeTerminal::Known) &&
        (success ? final.check.ok
                 : same_check(final.check, control->first_failure));
    if (!valid) {
      final.check = {false, "accel_kernel_pipeline_invalid"};
      final.terminal = NativeTerminal::UnknownMayWrite;
    }
    quarantine = !valid || control->quarantined ||
                 final.terminal == NativeTerminal::UnknownMayWrite;
    completion = control->bound.final;
    user = control->bound.user;
    for (std::size_t role = 0u; role < control->bound.role_count; ++role) {
      owners[role] = control->bound.roles[role].pipeline;
    }
    states = control->states;
    state_count = PreparedScheduleSortedStates(control->bound, states);
    PreparedResidencyStreamControl *const stream = control->stream;
    lock.unlock();
    std::array<std::unique_lock<std::mutex>, ResidencyScheduleRoleCapacity>
        claims{};
    for (std::size_t index = 0u; index < state_count; ++index) {
      claims[index] =
          std::unique_lock<std::mutex>{states[index]->submission.mutex};
    }
    std::unique_lock stream_lock{stream->gate};
    lock.lock();
    if (!control->active || control->stream != stream) {
      return;
    }
    for (std::size_t index = 0u; index < state_count; ++index) {
      prepared::PipelineSubmission &submission = states[index]->submission;
      if (submission.schedule == control) {
        submission.schedule = nullptr;
      }
      submission.quarantined = quarantine;
    }
    stream->quarantined = stream->quarantined || quarantine;
    control->bound = {};
    control->backend = {};
    control->states.fill(nullptr);
    control->stream = nullptr;
    control->active = false;
    control->aborting = false;
    control->quarantined = quarantine;
    lock.unlock();
  }
  completion(user,
             PreparedResidencyScheduleFinal{.evidence = std::move(final)});
  static_cast<void>(owners);
}


} // namespace rund::node::accel::detail

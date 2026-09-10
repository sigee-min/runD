#include "local.hpp"

#include "../../local.hpp"
#include "../../state.hpp"

#include "../../../stats.hpp"
#include "../../../status.hpp"

#include <chrono>
#include <mutex>
#include <utility>

namespace rund::compute::detail {
namespace {

[[nodiscard]] std::uint64_t now_ns() noexcept {
  return static_cast<std::uint64_t>(
      std::chrono::duration_cast<std::chrono::nanoseconds>(
          std::chrono::steady_clock::now().time_since_epoch())
          .count());
}

void clear_control(residency::execution::Control &control) noexcept {
  control.local_count = 0u;
  control.bound_plan = 0u;
  control.bound_token = 0u;
  control.bound_generation = 0u;
  control.bound_evidence = nullptr;
  control.bound_completion = nullptr;
  control.bound_user = nullptr;
  control.active.store(false, std::memory_order_release);
}

[[nodiscard]] residency::execution::NativeEvidence
native_evidence(const PipelineExecutionSubmission &submission,
                const node::accel::detail::PreparedPipelineEvidence &raw,
                const Status terminal) noexcept {
  const residency::execution::Control &control = *submission.control;
  const bool known = raw.terminal == node::accel::detail::NativeTerminal::Known;
  const std::uint64_t native_submissions =
      raw.shared.run.work.command_submit_count;
  const bool submitted = raw.submitted || native_submissions != 0u;
  residency::execution::NativeEvidence evidence{
      .status = terminal,
      .terminal = known ? residency::execution::TerminalKind::Known
                        : residency::execution::TerminalKind::UnknownMayWrite,
      .plan_identity = control.bound_plan,
      .token = control.bound_token,
      .generation = control.bound_generation,
      .epoch_count = 1u,
      .native_submissions = native_submissions,
      .native_dispatches = submitted ? 1u : 0u,
      .native_completions = submitted && known ? 1u : 0u,
      .native_inflight_peak = raw.shared.run.work.command_inflight_peak,
      .completed_ns = now_ns(),
  };
  if (!terminal) {
    constexpr std::uint8_t dispatch =
        std::uint8_t{1u} << static_cast<std::uint8_t>(
            residency::execution::Phase::Dispatch);
    evidence.failures[0u] = residency::execution::FailureEvidence{
        .epoch = 0u,
        .phases = dispatch,
        .may_write = submitted ? dispatch : std::uint8_t{0u},
        .terminal = submitted && known ? dispatch : std::uint8_t{0u},
    };
    evidence.failure_count = 1u;
  }
  return evidence;
}

void complete_pipeline_execution(
    void *const raw,
    node::accel::detail::PreparedPipelineEvidence &&evidence) noexcept {
  auto *const submission = static_cast<PipelineExecutionSubmission *>(raw);
  if (submission == nullptr || submission->control == nullptr ||
      submission->pipeline == nullptr) {
    return;
  }
  PipelineExecutionPhase expected = PipelineExecutionPhase::Submitting;
  if (!submission->phase.compare_exchange_strong(
          expected, PipelineExecutionPhase::Completed,
          std::memory_order_acq_rel, std::memory_order_acquire)) {
    expected = PipelineExecutionPhase::Submitted;
    if (!submission->phase.compare_exchange_strong(
            expected, PipelineExecutionPhase::Completed,
            std::memory_order_acq_rel, std::memory_order_acquire)) {
      return;
    }
  }

  residency::execution::Control &control = *submission->control;
  if (!control.active.load(std::memory_order_acquire) ||
      control.bound_plan == 0u || control.bound_token == 0u ||
      control.bound_generation == 0u || control.bound_evidence == nullptr ||
      control.bound_completion == nullptr || submission->issued_steps == 0u ||
      submission->issued_steps > submission->locals.size()) {
    return;
  }

  Status terminal = Status::fail(Reason::CompletionInvalid);
  {
    PipelineState &state = *submission->pipeline;
    std::lock_guard lock{state.gate};
    if (state.phase == PipelinePhase::Running) {
      PipelineOutcome outcome{
          .writes_possible = submission->writes_possible,
          .publication_suppressed = true,
      };
      outcome = finish_residency_pipeline_execution(
          state,
          std::span<const std::uint32_t>{submission->locals.data(),
                                         submission->issued_steps},
          evidence, outcome);
      terminal = publish_residency_pipeline_execution(
          state, submission->issued_steps, outcome);
    }
  }

  residency::execution::NativeEvidence projected =
      native_evidence(*submission, evidence, terminal);
  *control.bound_evidence = projected;
  const residency::execution::Completion completion = control.bound_completion;
  void *const user = control.bound_user;
  clear_control(control);
  completion(user, std::move(projected));
}

} // namespace

void clear_pipeline_execution_control(
    residency::execution::Control &control) noexcept {
  clear_control(control);
}

void complete_pipeline_execution_callback(
    void *const raw,
    node::accel::detail::PreparedPipelineEvidence &&evidence) noexcept {
  complete_pipeline_execution(raw, std::move(evidence));
}

} // namespace rund::compute::detail

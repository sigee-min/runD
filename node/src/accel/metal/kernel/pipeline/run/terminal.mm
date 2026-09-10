#include "../state.hpp"

#include "../residency/local.hpp"

#include "../../../../clock.hpp"
#include "../../../../kernel/reset/stats.hpp"
#include "../../../../kernel/telemetry.hpp"

#include <cstring>
#include <limits>
#include <utility>

namespace rund::node::accel::detail {

#if defined(__APPLE__) && defined(RUND_NODE_HAVE_METAL_SDK)
[[nodiscard]] bool
ObserveMetalControl(MetalSequence &sequence,
                    PreparedPipelineBackendEvidence &evidence) noexcept {
  if (sequence.control == nil || [sequence.control contents] == nullptr) {
    return false;
  }
  const std::uint64_t begin = MonotonicNanoseconds();
  std::memcpy(&evidence.control, [sequence.control contents],
              sizeof(evidence.control));
  const std::uint64_t observed_ns = MonotonicNanoseconds() - begin;
  evidence.control_ns =
      observed_ns >
              std::numeric_limits<std::uint64_t>::max() - evidence.control_ns
          ? std::numeric_limits<std::uint64_t>::max()
          : evidence.control_ns + observed_ns;
  evidence.control_command_count = sequence.control_command_count;
  evidence.submitted = true;
  evidence.control_observed = true;
  return true;
}

[[nodiscard]] bool
ObserveMetalProfile(MetalSequence &sequence,
                    PreparedPipelineBackendEvidence &evidence) noexcept {
  if (!sequence.profile_steps) {
    return true;
  }
  if (sequence.step_control == nil ||
      [sequence.step_control contents] == nullptr ||
      sequence.step_evidence.empty()) {
    return false;
  }
  const auto *const controls = static_cast<const PreparedPipelineStepControl *>(
      [sequence.step_control contents]);
  for (std::size_t index = 0u; index < sequence.step_evidence.size(); ++index) {
    PreparedPipelineStepEvidence &row = sequence.step_evidence[index];
    const bool visible =
        evidence.control.reason == 0u ||
        (evidence.control.failed_step != PreparedPipelineNoStep &&
         index <= evidence.control.failed_step);
    row.control = visible ? controls[index] : PreparedPipelineStepControl{};
    row.duration_ns = 0u;
    row.timing_sample_count = 0u;
    row.work_sample_count = visible ? 1u : 0u;
    row.clock = PreparedPipelineStepClock::Unavailable;
    row.relation = PreparedPipelineStepTimingRelation::Unavailable;
  }
  evidence.profile = PreparedPipelineProfileEvidence{
      .steps =
          std::span<const PreparedPipelineStepEvidence>{
              sequence.step_evidence.data(), sequence.step_evidence.size()},
      .instrumentation_command_count = 0u,
      .instrumentation_byte_count = sequence.instrumentation_byte_count,
      .observed = true,
  };
  return true;
}

bool ObserveMetalResidencyScheduleEvidence(
    MetalSequence &sequence,
    PreparedPipelineBackendEvidence &evidence) noexcept {
  return ObserveMetalControl(sequence, evidence) &&
         ObserveMetalProfile(sequence, evidence);
}

void CompleteMetalSequenceRun(void *const raw, KernelResult result,
                              const bool trace) noexcept {
  auto *const state = static_cast<submission::State<MetalSequence> *>(raw);
  if (state == nullptr) {
    return;
  }
  const submission::Claim<MetalSequence> claim = submission::Take(*state);
  if (!claim) {
    return;
  }
  MetalSequence *const sequence = claim.owner;
  if (trace) {
    if (result.check.ok) {
      result.check = FoldMetalDispatchTrace(
          sequence->trace,
          (__bridge id<MTLDevice>)sequence->adapter->device.get(),
          result.stats);
      if (result.check.ok) {
        RecordMetalDispatchTrace(*sequence->adapter,
                                 result.stats.run.time.accel_kernel_ns,
                                 result.stats.run.time.accel_timestamp_count);
      }
    }
  }
  result.pipeline.submitted = true;
  if (!claim.owner_local) {
    result.pipeline.control_command_count = sequence->control_command_count;
  }
  result.pipeline.control_ns = result.stats.run.time.command_submit_wait_ns;
  if (result.check.ok && !ObserveMetalControl(*sequence, result.pipeline)) {
    result.check = rund::AccelCheck{false, "accel_metal_buffer_unavailable"};
  }
  if (result.check.ok && !ObserveMetalProfile(*sequence, result.pipeline)) {
    result.check = rund::AccelCheck{false, "accel_metal_buffer_unavailable"};
  }
  if (result.check.ok) {
    ProjectTelemetry(result.pipeline.control, result.stats);
  }
  if (!claim.owner_local) {
    result.stats.run.work.dispatch_count =
        result.check.ok ? sequence->dispatch_count : 0u;
  } else if (!result.check.ok) {
    result.stats.run.work.dispatch_count = 0u;
  }
  if (!claim.owner_local) {
    SetResetStats(result.stats, result.check.ok, sequence->reset_count,
                  sequence->reset_bytes);
  } else if (!result.check.ok) {
    SetResetStats(result.stats, false, 0u, 0u);
  }
  if (result.check.ok) {
    RecordMetalDispatches(*sequence->adapter,
                          result.stats.run.work.dispatch_count);
  }
  claim.completion(claim.user, result);
}

void CompleteMetalSequence(void *const raw, KernelResult result) noexcept {
  CompleteMetalSequenceRun(raw, result, false);
}

#if defined(__MAC_26_0) && defined(__MAC_OS_X_VERSION_MAX_ALLOWED) &&          \
    __MAC_OS_X_VERSION_MAX_ALLOWED >= __MAC_26_0
void CompleteMetalResidencySubmission(MetalSequence &sequence,
                                      const std::uint64_t generation,
                                      const NativeTerminal terminal) noexcept {
  MetalResidencySubmission &submission = sequence.residency_submission;
  std::unique_lock terminal_gate{sequence.adapter->residency_terminal_gate};
  if (!submission.terminal.resolve(generation, terminal)) {
    return;
  }
  const bool sliding = sequence.residency_sliding.active;
  submission.watchdog_deadline_ns.store(0u, std::memory_order_release);
  dispatch_source_set_timer(submission.watchdog, DISPATCH_TIME_FOREVER,
                            DISPATCH_TIME_FOREVER, 0u);
  const bool native_terminal = terminal == NativeTerminal::Known;
  bool inflight_accounted = true;
  if (native_terminal) {
    [submission.allocator reset];
    inflight_accounted = EndMetalResidencyCommand(*sequence.adapter);
    if (!inflight_accounted) {
      sequence.adapter->residency_quarantined.store(true,
                                                    std::memory_order_release);
      submission.phase = MetalResidencySubmissionPhase::Quarantined;
    }
  } else {
    sequence.adapter->residency_quarantined.store(true,
                                                  std::memory_order_release);
    submission.phase = MetalResidencySubmissionPhase::Quarantined;
    SetMetalLastError(*sequence.adapter, "compute_device_lost");
  }
  const std::uint64_t submit_wait_ns =
      MonotonicNanoseconds() - submission.submit_begin_ns;
  KernelResult result{
      .check = rund::AccelCheck{true, "ok"},
      .stats =
          rund::RuntimeStats{
              .run =
                  {
                      .work =
                          {
                              .dispatch_count = submission.dispatch_count,
                              .command_submit_count =
                                  submission.native_submit_count,
                              .command_inflight_peak =
                                  submission.command_inflight_peak,
                              .reset_command_count = submission.reset_count,
                              .reset_bytes = submission.reset_bytes,
                          },
                      .time = {.command_submit_wait_ns =
                                   !native_terminal ||
                                           submission.native_submit_count == 0u
                                       ? 0u
                                       : submit_wait_ns},
                  },
              .outcome = {.ok = true, .reason = "ok"},
          },
      .pipeline = {.control_command_count = submission.control_count},
      .terminal = terminal,
  };
  if (native_terminal && submission.native_submit_count != 0u) {
    if (sliding) {
      RecordMetalCommandSubmitWaitOnlyNs(*sequence.adapter, submit_wait_ns);
    } else {
      RecordMetalCommandSubmitWaitNs(*sequence.adapter, submit_wait_ns);
    }
  }
  if (!native_terminal || !inflight_accounted || submission.force_device_lost) {
    SetMetalLastError(*sequence.adapter, "compute_device_lost");
    result.check = rund::AccelCheck{false, "compute_device_lost"};
  }
  result.stats.outcome.ok = result.check.ok;
  result.stats.outcome.reason = result.check.reason;
  if (!result.check.ok) {
    if (native_terminal) {
      submission.phase = MetalResidencySubmissionPhase::Failed;
    }
    result.stats.run.work.dispatch_count = 0u;
    result.stats.run.work.reset_command_count = 0u;
    result.stats.run.work.reset_bytes = 0u;
    result.pipeline.control_command_count = 0u;
  }
  if (native_terminal) {
    static_cast<void>(submission.terminal.release(generation, terminal));
  }
  std::shared_ptr<void> sliding_owner{};
  if (sliding) {
    sliding_owner = FinalizeMetalResidencySliding(sequence, result);
  }
  terminal_gate.unlock();
  CompleteMetalSequence(&sequence.submission, std::move(result));
  // Finalize cleared every shared gate field before the callback.  Only the
  // local strong retain is released afterward, so a reentrant next run cannot
  // be overwritten by cleanup from this generation.
  sliding_owner.reset();
}


#endif

void CompleteMetalSequenceTrace(void *const raw, KernelResult result) noexcept {
  CompleteMetalSequenceRun(raw, result, true);
}


#endif

} // namespace rund::node::accel::detail

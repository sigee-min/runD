#include "state.hpp"

#include "../../../kernel/reset/stats.hpp"
#include "../../../kernel/telemetry.hpp"
#include "../../pipeline/named.hpp"

#include <cstring>
#include <limits>

namespace rund::node::accel::detail {

#if defined(__APPLE__) && defined(RUND_NODE_HAVE_METAL_SDK)
[[nodiscard]] bool
ValidMetalSequence(const MetalSequence *const sequence) noexcept {
  return sequence != nullptr && sequence->adapter != nullptr &&
         ((sequence->command_count == 0u && sequence->command_chunks.empty()) ||
          (sequence->command_count != 0u && !sequence->command_chunks.empty() &&
           sequence->control != nil &&
           sequence->warm.owns(sequence->residency, sequence->command_chunks) &&
           (sequence->direct_aggregate || sequence->guard_zero != nil) &&
           (!sequence->direct_aggregate ||
            (sequence->command_count == 2u && sequence->state_count == 0u &&
             sequence->guard_zero == nil && sequence->states == nil &&
             sequence->raw_status == nil && !sequence->uses_status_arena)) &&
           (sequence->state_count == 0u || sequence->states != nil) &&
           (!sequence->profile_steps || (sequence->step_control != nil &&
                                         !sequence->step_evidence.empty())) &&
           (!sequence->uses_status_arena || sequence->raw_status != nil)));
}

[[nodiscard]] bool EmptyMetalSequence(const MetalSequence &sequence) noexcept {
  return sequence.command_count == 0u;
}

[[nodiscard]] rund::AccelCheck
EncodeMetalWarmSubmission(MetalAdapter &adapter, const MetalWarmSubmission warm,
                          CommandRun &command) {
  if (warm.chunks == nullptr || warm.chunk_count == 0u) {
    return rund::AccelCheck{false, "accel_kernel_pipeline_invalid"};
  }
  const rund::AccelCheck ready =
      OpenCommand<ResourceRefs::Borrowed>(adapter, command);
  if (!ready.ok) {
    return ready;
  }
  if (warm.resource_count != 0u) {
    [command.encoder useResources:warm.resources
                            count:warm.resource_count
                            usage:MTLResourceUsageRead | MTLResourceUsageWrite];
  }
  const rund::AccelCheck encoded = EncodeMetalPipelineIcbChunks(
      command.encoder, warm.chunks, warm.chunk_count);
  CloseCommand(command);
  return encoded;
}

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

void CompleteMetalSequence(void *const raw, KernelResult result) noexcept {
  auto *const state = static_cast<submission::State<MetalSequence> *>(raw);
  if (state == nullptr) {
    return;
  }
  const submission::Claim<MetalSequence> claim = submission::Take(*state);
  if (!claim) {
    return;
  }
  MetalSequence *const sequence = claim.owner;
  result.pipeline.submitted = true;
  result.pipeline.control_command_count = sequence->control_command_count;
  result.pipeline.control_ns = result.stats.command_submit_wait_ns;
  if (result.check.ok && !ObserveMetalControl(*sequence, result.pipeline)) {
    result.check = rund::AccelCheck{false, "accel_metal_buffer_unavailable"};
  }
  if (result.check.ok && !ObserveMetalProfile(*sequence, result.pipeline)) {
    result.check = rund::AccelCheck{false, "accel_metal_buffer_unavailable"};
  }
  if (result.check.ok) {
    ProjectTelemetry(result.pipeline.control, result.stats);
  }
  result.stats.dispatch_count = result.check.ok ? sequence->dispatch_count : 0u;
  SetResetStats(result.stats, result.check.ok, sequence->reset_count,
                sequence->reset_bytes);
  if (result.check.ok) {
    RecordMetalDispatches(*sequence->adapter, sequence->dispatch_count);
  }
  claim.completion(claim.user, result);
}

rund::AccelCheck
SeedPreparedMetalPipelineGeneration(const std::shared_ptr<void> &prepared,
                                    const std::uint32_t generation) noexcept {
  @autoreleasepool {
    auto *const pipeline = static_cast<MetalSequence *>(prepared.get());
    if (!ValidMetalSequence(pipeline)) {
      return rund::AccelCheck{false, "accel_kernel_pipeline_invalid"};
    }
    std::lock_guard lock{pipeline->submission.mutex};
    if (pipeline->submission.active()) {
      return rund::AccelCheck{false, "compute_pipeline_busy"};
    }
    if (EmptyMetalSequence(*pipeline)) {
      return rund::AccelCheck{true, "ok"};
    }
    void *const contents = [pipeline->control contents];
    if (contents == nullptr) {
      return rund::AccelCheck{false, "accel_metal_buffer_unavailable"};
    }
    const PreparedPipelineControl initial{.generation = generation};
    std::memcpy(contents, &initial, sizeof(initial));
    return rund::AccelCheck{true, "ok"};
  }
}

rund::AccelCheck
SubmitPreparedMetalPipeline(const std::shared_ptr<void> &prepared,
                            const KernelCompletion completion_fn,
                            void *const user) noexcept {
  @autoreleasepool {
    auto *const pipeline = static_cast<MetalSequence *>(prepared.get());
    if (!ValidMetalSequence(pipeline) || completion_fn == nullptr ||
        user == nullptr) {
      return rund::AccelCheck{false, "accel_kernel_run_invalid"};
    }
    if (EmptyMetalSequence(*pipeline)) {
      completion_fn(user, KernelResult{.check = rund::AccelCheck{true, "ok"},
                                       .stats = rund::RuntimeStats{
                                           .ok = true, .reason = "ok"}});
      return rund::AccelCheck{true, "ok"};
    }
    submission::State<MetalSequence> &state = pipeline->submission;
    if (!submission::Begin(state, *pipeline, completion_fn, user)) {
      return rund::AccelCheck{false, "compute_pipeline_busy"};
    }
    CommandRun command{};
    const rund::AccelCheck ready =
        EncodeMetalWarmSubmission(*pipeline->adapter, pipeline->warm, command);
    const rund::AccelCheck submitted =
        ready.ok
            ? QueueCommand(*pipeline->adapter, (__bridge void *)command.buffer,
                           CompleteMetalSequence, &state)
            : ready;
    if (!submitted.ok) {
      submission::Cancel(state);
    }
    return submitted;
  }
}

#endif

} // namespace rund::node::accel::detail

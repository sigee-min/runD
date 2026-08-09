#include "state.hpp"

#include "../trace/submit.hpp"

#include "../../../kernel/reset/stats.hpp"
#include "../../../kernel/telemetry.hpp"
#include "../../pipeline/named.hpp"

#include <cstring>
#include <limits>

namespace rund::node::accel::detail {

#if defined(__APPLE__) && defined(RUND_NODE_HAVE_METAL_SDK)
namespace {

[[nodiscard]] rund::AccelCheck
EnsureMetalPipelineDispatchTrace(MetalSequence &sequence) noexcept {
  if (sequence.trace.available()) {
    return rund::AccelCheck{true, "ok"};
  }
  if (sequence.adapter != nullptr &&
      sequence.adapter->fault_trace_unavailable_once.exchange(
          false, std::memory_order_relaxed)) {
    return rund::AccelCheck{false, "compute_telemetry_trace_unavailable"};
  }
  if (sequence.memory_meter == nullptr || sequence.adapter == nullptr) {
    return rund::AccelCheck{false, "compute_telemetry_trace_unavailable"};
  }
  MetalDispatchTrace candidate{};
  PrepareMetalDispatchTrace(
      (__bridge id<MTLDevice>)sequence.adapter->device.get(),
      sequence.dispatch_count, candidate);
  if (!candidate.available()) {
    return rund::AccelCheck{false, candidate.reason};
  }
  const std::uint64_t retained = candidate.retained_bytes();
  sequence.trace = std::move(candidate);
  sequence.memory_meter->add_device(PreparedMemory{.current = retained,
                                                   .peak = retained,
                                                   .cumulative = retained,
                                                   .budget = retained});
  sequence.retained_bytes =
      ::rund::detail::counter::SaturatingAdd(sequence.retained_bytes, retained);
  return rund::AccelCheck{true, "ok"};
}

} // namespace

[[nodiscard]] bool
ValidMetalSequence(const MetalSequence *const sequence) noexcept {
  return sequence != nullptr && sequence->adapter != nullptr &&
         ((sequence->command_count == 0u && sequence->command_chunks.empty()) ||
          (sequence->command_count != 0u && !sequence->command_chunks.empty() &&
           sequence->trace_commands.size() == sequence->dispatch_count &&
           !sequence->trace_commands.empty() &&
           sequence->trace_commands.back() < sequence->command_count &&
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
                          const std::span<const std::uint64_t> trace_commands,
                          CommandRun &command,
                          MetalDispatchTrace *const trace) {
  if (warm.chunks == nullptr || warm.chunk_count == 0u) {
    return rund::AccelCheck{false, "accel_kernel_pipeline_invalid"};
  }
  if (trace != nullptr &&
      trace->sampling == MetalTraceSampling::StageBoundary) {
    id<MTLCommandQueue> const queue =
        (__bridge id<MTLCommandQueue>)adapter.queue.get();
    command.buffer =
        queue == nil ? nil : [queue commandBufferWithUnretainedReferences];
    if (command.buffer == nil) {
      return rund::AccelCheck{false, "accel_metal_command_unavailable"};
    }
    BeginMetalDispatchTrace((__bridge id<MTLDevice>)adapter.device.get(),
                            *trace);
    std::uint64_t global_command = 0u;
    std::size_t trace_index = 0u;
    for (NSUInteger chunk_index = 0u; chunk_index < warm.chunk_count;
         ++chunk_index) {
      const MetalIcbChunk &chunk = warm.chunks[chunk_index];
      if (!chunk.valid() || (chunk_index == 0u && chunk.barrier_before())) {
        trace->failed = true;
        return rund::AccelCheck{false, "accel_kernel_pipeline_invalid"};
      }
      for (NSUInteger index = 0u; index < chunk.command_count; ++index) {
        const bool sampled = trace_index < trace_commands.size() &&
                             trace_commands[trace_index] == global_command;
        id<MTLComputeCommandEncoder> const encoder =
            sampled ? OpenMetalTraceStageEncoder(command.buffer, *trace)
                    : [command.buffer computeCommandEncoder];
        if (encoder == nil) {
          return rund::AccelCheck{false, "compute_telemetry_trace_unavailable"};
        }
        if (warm.resource_count != 0u) {
          [encoder useResources:warm.resources
                          count:warm.resource_count
                          usage:MTLResourceUsageRead | MTLResourceUsageWrite];
        }
        [encoder executeCommandsInBuffer:chunk.commands
                               withRange:NSMakeRange(index, 1u)];
        [encoder endEncoding];
        if (sampled) {
          ++trace_index;
        }
        ++global_command;
      }
    }
    return trace_index == trace_commands.size() &&
                   SealMetalDispatchTrace(command.buffer, *trace)
               ? rund::AccelCheck{true, "ok"}
               : rund::AccelCheck{false, "compute_telemetry_trace_unavailable"};
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
  if (trace != nullptr) {
    BeginMetalDispatchTrace((__bridge id<MTLDevice>)adapter.device.get(),
                            *trace);
  }
  const rund::AccelCheck encoded =
      trace == nullptr
          ? EncodeMetalPipelineIcbChunks(command.encoder, warm.chunks,
                                         warm.chunk_count)
          : EncodeMetalPipelineIcbTrace(command.encoder, warm.chunks,
                                        warm.chunk_count, trace_commands,
                                        *trace);
  CloseCommand(command);
  if (!encoded.ok || trace == nullptr) {
    return encoded;
  }
  return SealMetalDispatchTrace(command.buffer, *trace)
             ? encoded
             : rund::AccelCheck{false, "compute_telemetry_trace_unavailable"};
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
  result.pipeline.control_command_count = sequence->control_command_count;
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
  result.stats.run.work.dispatch_count =
      result.check.ok ? sequence->dispatch_count : 0u;
  SetResetStats(result.stats, result.check.ok, sequence->reset_count,
                sequence->reset_bytes);
  if (result.check.ok) {
    RecordMetalDispatches(*sequence->adapter, sequence->dispatch_count);
  }
  claim.completion(claim.user, result);
}

void CompleteMetalSequence(void *const raw, KernelResult result) noexcept {
  CompleteMetalSequenceRun(raw, result, false);
}

void CompleteMetalSequenceTrace(void *const raw, KernelResult result) noexcept {
  CompleteMetalSequenceRun(raw, result, true);
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

rund::AccelCheck SubmitPreparedMetalPipeline(
    const std::shared_ptr<void> &prepared, const KernelCompletion completion_fn,
    void *const user, const KernelTiming timing) noexcept {
  @autoreleasepool {
    auto *const pipeline = static_cast<MetalSequence *>(prepared.get());
    if (!ValidMetalSequence(pipeline) || completion_fn == nullptr ||
        user == nullptr) {
      return rund::AccelCheck{false, "accel_kernel_run_invalid"};
    }
    if (EmptyMetalSequence(*pipeline)) {
      completion_fn(user,
                    KernelResult{.check = rund::AccelCheck{true, "ok"},
                                 .stats = rund::RuntimeStats{
                                     .outcome = {.ok = true, .reason = "ok"}}});
      return rund::AccelCheck{true, "ok"};
    }
    submission::State<MetalSequence> &state = pipeline->submission;
    {
      std::lock_guard lock{state.mutex};
      if (state.active()) {
        return rund::AccelCheck{false, "compute_pipeline_busy"};
      }
      if (timing == KernelTiming::Dispatch) {
        const rund::AccelCheck traced =
            EnsureMetalPipelineDispatchTrace(*pipeline);
        if (!traced.ok) {
          return traced;
        }
      }
      state.owner = pipeline;
      state.completion = completion_fn;
      state.user = user;
    }
    CommandRun command{};
    MetalDispatchTrace *const trace =
        timing == KernelTiming::Dispatch ? &pipeline->trace : nullptr;
    const rund::AccelCheck ready =
        EncodeMetalWarmSubmission(*pipeline->adapter, pipeline->warm,
                                  pipeline->trace_commands, command, trace);
    rund::AccelCheck submitted = ready;
    if (ready.ok) {
      // Queue publication and terminal Take share this gate, making every
      // encode-side host write visible before completion observes resources.
      std::lock_guard lock{state.mutex};
      submitted = trace != nullptr
                      ? QueueMetalDispatchTrace(
                            *pipeline->adapter, command.buffer, *trace,
                            CompleteMetalSequenceTrace, &state)
                      : QueueCommand(*pipeline->adapter,
                                     (__bridge void *)command.buffer,
                                     CompleteMetalSequence, &state,
                                     timing == KernelTiming::Submission);
    }
    if (!submitted.ok) {
      submission::Cancel(state);
    }
    return submitted;
  }
}

#endif

} // namespace rund::node::accel::detail

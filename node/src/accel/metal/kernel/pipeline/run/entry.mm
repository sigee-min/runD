#include "../state.hpp"
#include "internal.hpp"

#include "../residency/local.hpp"

#include "../../trace/submit.hpp"

#include <cstring>
#include <utility>

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
    void *const user, const KernelTiming timing, const PipelineSubmitMode mode,
    const std::span<const std::uint32_t> locals) noexcept {
  @autoreleasepool {
    auto *const pipeline = static_cast<MetalSequence *>(prepared.get());
    if (!ValidMetalSequence(pipeline) || completion_fn == nullptr ||
        user == nullptr) {
      return rund::AccelCheck{false, "accel_kernel_run_invalid"};
    }
    const bool residency_mode = mode == PipelineSubmitMode::Residency;
    if ((residency_mode &&
         (timing != KernelTiming::Submission || locals.empty() ||
          !pipeline->residency_submission.ready())) ||
        (!residency_mode && !locals.empty())) {
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
      state.owner_local = residency_mode;
    }
    if (residency_mode) {
      const rund::AccelCheck submitted =
          SubmitMetalResidencySubmission(*pipeline, timing, locals);
      if (!submitted.ok) {
        submission::Cancel(state);
        return submitted;
      }
      return rund::AccelCheck{true, "ok"};
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

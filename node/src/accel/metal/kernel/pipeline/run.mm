#include "state.hpp"

#include "../../../kernel/reset/stats.hpp"
#include "../../../kernel/telemetry.hpp"
#include "../../pipeline/named.hpp"

#include <algorithm>
#include <cstring>
#include <limits>

namespace rund::node::accel::detail {

#if defined(__APPLE__) && defined(RUND_NODE_HAVE_METAL_SDK)
[[nodiscard]] bool
ValidMetalSequence(const MetalSequence *const sequence) noexcept {
  return sequence != nullptr && sequence->adapter != nullptr &&
         ((sequence->command_count == 0u && sequence->commands == nil) ||
          (sequence->command_count != 0u &&
           (sequence->commands != nil || !sequence->direct.empty()) &&
           sequence->control != nil &&
           ((sequence->range_capacity == 0u && sequence->range_buffer == nil &&
             sequence->range_owners == nil) ||
            (sequence->range_capacity != 0u && sequence->range_buffer != nil &&
             sequence->range_owners != nil)) &&
           sequence->ranges.size() <= sequence->range_capacity &&
           sequence->ranges.size() == sequence->original_ranges.size() &&
           (sequence->state_count == 0u || sequence->states != nil) &&
           (sequence->gate_count == 0u ||
            (sequence->gate != nil && sequence->gate_buffer != nil &&
             sequence->states != nil)) &&
           (!sequence->profile_steps || (sequence->step_control != nil &&
                                         !sequence->step_evidence.empty())) &&
           (!sequence->uses_status_arena || sequence->raw_status != nil)));
}

[[nodiscard]] bool EmptyMetalSequence(const MetalSequence &sequence) noexcept {
  return sequence.command_count == 0u;
}

[[nodiscard]] rund::AccelCheck EncodeMetalSequence(MetalSequence &sequence,
                                                   CommandRun &command) {
  if (EmptyMetalSequence(sequence)) {
    return rund::AccelCheck{true, "ok"};
  }
  const rund::AccelCheck ready =
      OpenCommand<ResourceRefs::Borrowed>(*sequence.adapter, command);
  if (!ready.ok) {
    return ready;
  }
  if (!sequence.declared.empty()) {
    [command.encoder useResources:sequence.declared.data()
                            count:sequence.declared.size()
                            usage:MTLResourceUsageRead | MTLResourceUsageWrite];
  }
  std::size_t range_index = 0u;
  bool ranges_valid = true;
  const auto execute_fixed = [&](NSUInteger begin, const NSUInteger end) {
    bool visible_end = false;
    while (range_index < sequence.ranges.size()) {
      const MetalRangePlan &plan = sequence.ranges[range_index];
      const std::uint64_t range_end =
          static_cast<std::uint64_t>(plan.range.location) + plan.range.length;
      if (plan.range.location >= end) {
        break;
      }
      if (plan.range.location < begin || range_end > end) {
        ranges_valid = false;
        break;
      }
      [command.encoder
          executeCommandsInBuffer:sequence.commands
                   indirectBuffer:sequence.range_buffer
             indirectBufferOffset:range_index * sizeof(MetalRange)];
      visible_end = plan.barrier && range_end == end;
      if (plan.barrier) {
        [command.encoder memoryBarrierWithScope:MTLBarrierScopeBuffers];
      }
      begin = static_cast<NSUInteger>(range_end);
      ++range_index;
    }
    return visible_end;
  };
  NSUInteger fixed_begin = 0u;
  for (const MetalCommand &source : sequence.direct) {
    const NSUInteger stream_index =
        static_cast<NSUInteger>(source.stream_index);
    if (stream_index > fixed_begin) {
      if (!execute_fixed(fixed_begin, stream_index)) {
        [command.encoder memoryBarrierWithScope:MTLBarrierScopeBuffers];
      }
    }
    const bool gated =
        source.kind == MetalGrid::IndirectGroups &&
        source.owner != std::numeric_limits<std::uint32_t>::max();
    if (gated) {
      [command.encoder setComputePipelineState:sequence.gate];
      [command.encoder setBuffer:source.indirect_buffer
                          offset:source.indirect_offset
                         atIndex:0u];
      [command.encoder setBuffer:sequence.gate_buffer
                          offset:source.gate_offset
                         atIndex:1u];
      [command.encoder setBuffer:sequence.states offset:0u atIndex:2u];
      [command.encoder setBytes:&source.owner
                         length:sizeof(source.owner)
                        atIndex:3u];
      [command.encoder dispatchThreads:MTLSizeMake(1u, 1u, 1u)
                 threadsPerThreadgroup:MTLSizeMake(1u, 1u, 1u)];
      [command.encoder memoryBarrierWithScope:MTLBarrierScopeBuffers];
    }
    [command.encoder setComputePipelineState:source.pipeline];
    const std::size_t binding_end = source.binding_begin + source.binding_count;
    for (std::size_t binding = source.binding_begin; binding < binding_end;
         ++binding) {
      const MetalCommandBinding &argument = sequence.indirect_bindings[binding];
      const bool uses_parameter = argument.buffer == nil;
      [command.encoder
          setBuffer:(uses_parameter ? sequence.parameters : argument.buffer)
             offset:(uses_parameter ? argument.parameter
                                    : argument.offset)atIndex:argument.index];
    }
    const std::size_t threadgroup_end =
        source.threadgroup_begin + source.threadgroup_count;
    for (std::size_t binding = source.threadgroup_begin;
         binding < threadgroup_end; ++binding) {
      const MetalThreadgroupBinding &argument =
          sequence.indirect_threadgroups[binding];
      [command.encoder setThreadgroupMemoryLength:argument.length
                                          atIndex:argument.index];
    }
    if (source.kind == MetalGrid::IndirectGroups) {
      [command.encoder
          dispatchThreadgroupsWithIndirectBuffer:(gated
                                                      ? sequence.gate_buffer
                                                      : source.indirect_buffer)
                            indirectBufferOffset:(gated
                                                      ? source.gate_offset
                                                      : source.indirect_offset)
                                                 threadsPerThreadgroup
                                                :source.threads];
    } else {
      [command.encoder dispatchThreads:source.grid
                 threadsPerThreadgroup:source.threads];
    }
    if (source.barrier) {
      [command.encoder memoryBarrierWithScope:MTLBarrierScopeBuffers];
    }
    fixed_begin = stream_index + 1u;
  }
  if (fixed_begin < sequence.command_count) {
    execute_fixed(fixed_begin, sequence.command_count);
  }
  if (!ranges_valid || range_index != sequence.ranges.size()) {
    CloseCommand(command);
    return rund::AccelCheck{false, "accel_kernel_pipeline_invalid"};
  }
  CloseCommand(command);
  return rund::AccelCheck{true, "ok"};
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
    if (pipeline->profile_steps) {
      auto *const controls = static_cast<PreparedPipelineStepControl *>(
          [pipeline->step_control contents]);
      if (controls == nullptr) {
        return rund::AccelCheck{false, "accel_metal_buffer_unavailable"};
      }
      std::fill_n(controls, pipeline->step_evidence.size(),
                  PreparedPipelineStepControl{});
    }
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
    if (pipeline->state_count != 0u) {
      void *const states = [pipeline->states contents];
      if (states == nullptr) {
        submission::Cancel(state);
        return rund::AccelCheck{false, "accel_metal_buffer_unavailable"};
      }
      std::memset(states, 0,
                  static_cast<std::size_t>(pipeline->state_count) *
                      sizeof(ResidentState));
    }
    void *const ranges = [pipeline->range_buffer contents];
    if ((!pipeline->original_ranges.empty() && ranges == nullptr) ||
        pipeline->original_ranges.size() >
            std::numeric_limits<std::size_t>::max() / sizeof(MetalRange)) {
      submission::Cancel(state);
      return rund::AccelCheck{false, "accel_metal_buffer_unavailable"};
    }
    if (!pipeline->original_ranges.empty()) {
      std::memcpy(ranges, pipeline->original_ranges.data(),
                  pipeline->original_ranges.size() * sizeof(MetalRange));
    }
    CommandRun command{};
    const rund::AccelCheck ready = EncodeMetalSequence(*pipeline, command);
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

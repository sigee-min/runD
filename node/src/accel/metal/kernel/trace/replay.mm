#include "replay.hpp"

#include "../../pipeline/named.hpp"
#include "../pipeline/aggregate/model.hpp"
#include "../pipeline/capture.hpp"
#include "encoder.hpp"

#include <kernel/core/checked.hpp>

#include <limits>

namespace rund::node::accel::detail {

#if defined(__APPLE__) && defined(RUND_NODE_HAVE_METAL_SDK)
namespace {

struct MetalTraceReplay final {
  MetalCapture capture{};
  id<MTLBuffer> parameters = nil;
};

[[nodiscard]] bool add(const std::uint64_t left, const std::uint64_t right,
                       std::uint64_t &value) noexcept {
  return rund::kernel::checked::add(left, right, value);
}

[[nodiscard]] rund::AccelCheck
BuildMetalTraceReplay(id<MTLDevice> const device, MetalAdapter &adapter,
                      MetalKernelResources &resources,
                      std::shared_ptr<void> &published,
                      std::uint64_t &retained) noexcept {
  published.reset();
  retained = 0u;
  if (device == nil || resources.dispatch_count == 0u ||
      resources.dispatch_count > std::numeric_limits<std::size_t>::max()) {
    return rund::AccelCheck{false, "compute_pipeline_capacity"};
  }
  std::uint64_t binding_capacity = 0u;
  std::uint64_t parameter_capacity = 0u;
  constexpr std::uint64_t parameter_bytes_per_dispatch =
      sizeof(MetalNestedAggregateParams);
  if (!rund::kernel::checked::mul(
          resources.dispatch_count,
          static_cast<std::uint64_t>(kMetalArgumentCapacity),
          binding_capacity) ||
      !rund::kernel::checked::mul(resources.dispatch_count,
                                  parameter_bytes_per_dispatch,
                                  parameter_capacity) ||
      binding_capacity > std::numeric_limits<std::size_t>::max() ||
      parameter_capacity > std::numeric_limits<std::size_t>::max()) {
    return rund::AccelCheck{false, "compute_pipeline_capacity"};
  }
  std::shared_ptr<MetalTraceReplay> replay;
  try {
    replay = std::make_shared<MetalTraceReplay>();
    MetalCapture &capture = replay->capture;
    capture.command_capacity =
        static_cast<std::size_t>(resources.dispatch_count);
    capture.binding_capacity = static_cast<std::size_t>(binding_capacity);
    capture.parameter_capacity = static_cast<std::size_t>(parameter_capacity);
    capture.producer_binding_slot_upper = kMetalArgumentCapacity;
    capture.unguarded = true;
    capture.commands.reserve(capture.command_capacity);
    capture.command_bindings.reserve(capture.binding_capacity);
    capture.parameters.reserve(capture.parameter_capacity);
    if (capture.commands.capacity() != capture.command_capacity ||
        capture.command_bindings.capacity() != capture.binding_capacity ||
        capture.parameters.capacity() != capture.parameter_capacity) {
      return rund::AccelCheck{false, "compute_pipeline_capacity"};
    }
  } catch (...) {
    return rund::AccelCheck{false, "compute_pipeline_capacity"};
  }
  RUNDMetalPipelineCapture *const encoder =
      [[RUNDMetalPipelineCapture alloc] initWithCapture:&replay->capture];
  if (encoder == nil) {
    return rund::AccelCheck{false, "compute_device_capacity"};
  }
  CommandRun command{.buffer = nil,
                     .encoder = (id<MTLComputeCommandEncoder>)encoder};
  const rund::AccelCheck encoded =
      EncodeMetalSteps(adapter, resources, command);
  if (!encoded.ok) {
    return encoded;
  }
  const rund::AccelCheck captured = CheckMetalPipelineCapture(replay->capture);
  if (!captured.ok) {
    return captured;
  }
  if (replay->capture.commands.size() != resources.dispatch_count) {
    return rund::AccelCheck{false, "compute_telemetry_trace_unavailable"};
  }
  if (!replay->capture.parameters.empty()) {
    replay->parameters =
        [device newBufferWithBytes:replay->capture.parameters.data()
                            length:replay->capture.parameters.size()
                           options:MTLResourceStorageModeShared];
    if (replay->parameters == nil || [replay->parameters contents] == nullptr) {
      return rund::AccelCheck{false, "compute_device_capacity"};
    }
  }
  std::uint64_t host = sizeof(MetalTraceReplay);
  std::uint64_t rows = 0u;
  if (!rund::kernel::checked::mul(
          static_cast<std::uint64_t>(replay->capture.commands.capacity()),
          sizeof(MetalCommand), rows) ||
      !add(host, rows, host) ||
      !rund::kernel::checked::mul(
          static_cast<std::uint64_t>(
              replay->capture.command_bindings.capacity()),
          sizeof(MetalCommandBinding), rows) ||
      !add(host, rows, host) ||
      !add(host,
           static_cast<std::uint64_t>(replay->capture.parameters.capacity()),
           host)) {
    return rund::AccelCheck{false, "compute_pipeline_capacity"};
  }
  const std::uint64_t device_bytes =
      replay->parameters == nil
          ? 0u
          : static_cast<std::uint64_t>([replay->parameters allocatedSize]);
  if (!add(host, device_bytes, retained)) {
    return rund::AccelCheck{false, "compute_pipeline_capacity"};
  }
  published = std::move(replay);
  return rund::AccelCheck{true, "ok"};
}

} // namespace

rund::AccelCheck
EnsureMetalPreparedDispatchTrace(id<MTLDevice> const device,
                                 MetalKernelResources &resources,
                                 PreparedMemoryMeter *const memory) noexcept {
  if (resources.trace.available() &&
      ((resources.trace.sampling == MetalTraceSampling::DispatchBoundary &&
        resources.trace_encoder != nil) ||
       (resources.trace.sampling == MetalTraceSampling::StageBoundary &&
        resources.trace_recipe != nullptr))) {
    return rund::AccelCheck{true, "ok"};
  }
  if (resources.adapter != nullptr &&
      resources.adapter->fault_trace_unavailable_once.exchange(
          false, std::memory_order_relaxed)) {
    return rund::AccelCheck{false, "compute_telemetry_trace_unavailable"};
  }
  MetalDispatchTrace candidate{};
  PrepareMetalDispatchTrace(device, resources.dispatch_count, candidate);
  if (!candidate.available()) {
    return rund::AccelCheck{false, candidate.reason};
  }
  if (memory == nullptr) {
    return rund::AccelCheck{false, "compute_device_capacity"};
  }
  id encoder = nil;
  std::shared_ptr<void> replay;
  std::uint64_t replay_retained = 0u;
  if (candidate.sampling == MetalTraceSampling::DispatchBoundary) {
    encoder = CreateMetalDispatchTraceEncoder();
    if (encoder == nil) {
      return rund::AccelCheck{false, "compute_device_capacity"};
    }
  } else {
    const rund::AccelCheck captured = BuildMetalTraceReplay(
        device, *resources.adapter, resources, replay, replay_retained);
    if (!captured.ok) {
      return captured;
    }
  }
  std::uint64_t retained = 0u;
  if (!add(candidate.retained_bytes(), replay_retained, retained)) {
    return rund::AccelCheck{false, "compute_pipeline_capacity"};
  }
  resources.trace = std::move(candidate);
  resources.trace_recipe = std::move(replay);
  resources.trace_encoder = encoder;
  memory->add(PreparedMemory{.current = retained,
                             .peak = retained,
                             .cumulative = retained,
                             .budget = retained});
  return rund::AccelCheck{true, "ok"};
}

rund::AccelCheck EncodeMetalPreparedStageTrace(MetalAdapter &adapter,
                                               MetalKernelResources &resources,
                                               id<MTLDevice> const device,
                                               CommandRun &run) noexcept {
  auto *const replay =
      static_cast<MetalTraceReplay *>(resources.trace_recipe.get());
  id<MTLCommandQueue> const queue =
      (__bridge id<MTLCommandQueue>)adapter.queue.get();
  run.buffer =
      queue == nil ? nil : [queue commandBufferWithUnretainedReferences];
  if (replay == nullptr || run.buffer == nil ||
      replay->capture.commands.size() != resources.dispatch_count) {
    return rund::AccelCheck{false, "compute_telemetry_trace_unavailable"};
  }
  BeginMetalDispatchTrace(device, resources.trace);
  for (const MetalCommand &record : replay->capture.commands) {
    id<MTLComputeCommandEncoder> const encoder =
        OpenMetalTraceStageEncoder(run.buffer, resources.trace);
    if (encoder == nil || record.pipeline == nil ||
        record.binding_begin > replay->capture.command_bindings.size() ||
        record.binding_count >
            replay->capture.command_bindings.size() - record.binding_begin) {
      return rund::AccelCheck{false, "compute_telemetry_trace_unavailable"};
    }
    [encoder setComputePipelineState:record.pipeline];
    const std::size_t end = record.binding_begin + record.binding_count;
    for (std::size_t index = record.binding_begin; index < end; ++index) {
      const MetalCommandBinding &binding =
          replay->capture.command_bindings[index];
      id<MTLBuffer> const buffer =
          binding.buffer == nil ? replay->parameters : binding.buffer;
      const NSUInteger offset = binding.buffer == nil
                                    ? static_cast<NSUInteger>(binding.parameter)
                                    : binding.offset;
      if (buffer == nil) {
        [encoder endEncoding];
        return rund::AccelCheck{false, "compute_telemetry_trace_unavailable"};
      }
      [encoder setBuffer:buffer offset:offset atIndex:binding.index];
    }
    switch (record.kind) {
    case MetalGrid::Groups:
      [encoder dispatchThreadgroups:record.grid
              threadsPerThreadgroup:record.threads];
      break;
    case MetalGrid::Threads:
      [encoder dispatchThreads:record.grid
          threadsPerThreadgroup:record.threads];
      break;
    case MetalGrid::Indirect:
      [encoder dispatchThreadgroupsWithIndirectBuffer:record.indirect
                                 indirectBufferOffset:record.indirect_offset
                                threadsPerThreadgroup:record.threads];
      break;
    case MetalGrid::None:
      resources.trace.failed = true;
      break;
    }
    [encoder endEncoding];
    if (resources.trace.failed) {
      return rund::AccelCheck{false, "compute_telemetry_trace_unavailable"};
    }
  }
  return SealMetalDispatchTrace(run.buffer, resources.trace)
             ? rund::AccelCheck{true, "ok"}
             : rund::AccelCheck{false, "compute_telemetry_trace_unavailable"};
}
#endif

} // namespace rund::node::accel::detail

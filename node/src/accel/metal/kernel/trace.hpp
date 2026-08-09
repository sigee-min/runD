#pragma once

#include <accel/check.hpp>
#include <accel/runtime.hpp>

#include "../../kernel/callback.hpp"

#include <cstdint>

#if defined(__OBJC__) && defined(__APPLE__) && defined(RUND_NODE_HAVE_METAL_SDK)
#import <Metal/Metal.h>

namespace rund::node::accel::detail {

struct MetalAdapter;

enum class MetalTraceSampling : std::uint8_t {
  None,
  DispatchBoundary,
  StageBoundary,
};

struct MetalDispatchTrace final {
  id<MTLCounterSampleBuffer> samples = nil;
  id<MTLBuffer> values = nil;
  MTLComputePassDescriptor *pass = nil;
  NSUInteger sample_count{};
  NSUInteger cursor{};
  MTLTimestamp cpu_start{};
  MTLTimestamp gpu_start{};
  MetalAdapter *submit_adapter{};
  KernelCompletion submit_completion{};
  void *submit_user{};
  const char *reason{"compute_telemetry_trace_unavailable"};
  MetalTraceSampling sampling{MetalTraceSampling::None};
  bool failed{};

  [[nodiscard]] bool available() const noexcept;
  [[nodiscard]] std::uint64_t retained_bytes() const noexcept;
};

void PrepareMetalDispatchTrace(id<MTLDevice> device,
                               std::uint64_t dispatch_count,
                               MetalDispatchTrace &trace) noexcept;
void BeginMetalDispatchTrace(id<MTLDevice> device,
                             MetalDispatchTrace &trace) noexcept;
void SampleMetalDispatchTrace(id<MTLComputeCommandEncoder> encoder,
                              MetalDispatchTrace &trace) noexcept;
[[nodiscard]] id<MTLComputeCommandEncoder>
OpenMetalTraceStageEncoder(id<MTLCommandBuffer> command,
                           MetalDispatchTrace &trace) noexcept;
[[nodiscard]] bool SealMetalDispatchTrace(id<MTLCommandBuffer> command,
                                          MetalDispatchTrace &trace) noexcept;
[[nodiscard]] rund::AccelCheck
FoldMetalDispatchTrace(MetalDispatchTrace &trace, id<MTLDevice> device,
                       rund::RuntimeStats &stats) noexcept;

} // namespace rund::node::accel::detail
#endif

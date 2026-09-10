#include "internal.hpp"

#include <algorithm>
#include <limits>

namespace rund::node::accel::detail {

#if defined(__APPLE__) && defined(RUND_NODE_HAVE_METAL_SDK)
namespace {

struct MetalViewParams final {
  std::uint64_t count{};
  std::uint64_t offset_words{};
  std::uint64_t stride_words{};
  std::uint32_t element_words{};
  std::uint32_t reserved{};
};

static_assert(sizeof(MetalViewParams) == 32u);

[[nodiscard]] rund::AccelCheck
EncodeMetalViewTransfers(const MetalViewLowering &view,
                         id<MTLComputeCommandEncoder> encoder,
                         const bool inputs) {
  if (encoder == nil) {
    return rund::AccelCheck{false, "accel_metal_command_unavailable"};
  }
  id<MTLComputePipelineState> pipeline =
      (__bridge id<MTLComputePipelineState>)(inputs
                                                 ? view.gather_pipeline.get()
                                                 : view.scatter_pipeline.get());
  bool encoded = false;
  for (const MetalViewTransfer &transfer : view.transfers) {
    if (transfer.input != inputs) {
      continue;
    }
    if (pipeline == nil || transfer.external.device_buffer == nullptr ||
        transfer.dense.device_buffer == nullptr ||
        (transfer.element_bytes != 4u && transfer.element_bytes != 8u) ||
        transfer.offset_bytes % sizeof(std::uint32_t) != 0u ||
        transfer.stride_bytes % sizeof(std::uint32_t) != 0u ||
        transfer.count > static_cast<std::uint64_t>(
                             std::numeric_limits<std::uint32_t>::max())) {
      return rund::AccelCheck{false, "compute_resident_stride_invalid"};
    }
    const MetalViewParams params{
        .count = transfer.count,
        .offset_words = transfer.offset_bytes / sizeof(std::uint32_t),
        .stride_words = transfer.stride_bytes / sizeof(std::uint32_t),
        .element_words = static_cast<std::uint32_t>(transfer.element_bytes /
                                                    sizeof(std::uint32_t)),
    };
    id<MTLBuffer> external =
        (__bridge id<MTLBuffer>)transfer.external.device_buffer.get();
    id<MTLBuffer> dense =
        (__bridge id<MTLBuffer>)transfer.dense.device_buffer.get();
    const NSUInteger dense_offset =
        static_cast<NSUInteger>(transfer.dense.ref.offset_bytes);
    [encoder setComputePipelineState:pipeline];
    [encoder setBuffer:(inputs ? external : dense)
                offset:(inputs ? 0u : dense_offset)atIndex:0u];
    [encoder setBuffer:(inputs ? dense : external)
                offset:(inputs ? dense_offset : 0u)atIndex:1u];
    [encoder setBytes:&params length:sizeof(params) atIndex:2u];
    const NSUInteger count = static_cast<NSUInteger>(transfer.count);
    [encoder dispatchThreads:MTLSizeMake(count, 1u, 1u)
        threadsPerThreadgroup:MTLSizeMake(std::min<NSUInteger>(count, 256u), 1u,
                                          1u)];
    encoded = true;
  }
  const bool closes_input_lifetime = !inputs && view.has_input;
  if (encoded || closes_input_lifetime) {
    [encoder memoryBarrierWithScope:MTLBarrierScopeBuffers];
  }
  return rund::AccelCheck{true, "ok"};
}

} // namespace

rund::AccelCheck
EncodeMetalViewInputs(const std::shared_ptr<MetalViewLowering> &view,
                      id<MTLComputeCommandEncoder> encoder) {
  return view == nullptr ? rund::AccelCheck{true, "ok"}
                         : EncodeMetalViewTransfers(*view, encoder, true);
}

rund::AccelCheck
EncodeMetalViewOutputs(const std::shared_ptr<MetalViewLowering> &view,
                       id<MTLComputeCommandEncoder> encoder) {
  return view == nullptr ? rund::AccelCheck{true, "ok"}
                         : EncodeMetalViewTransfers(*view, encoder, false);
}

#endif

} // namespace rund::node::accel::detail

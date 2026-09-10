#include "internal.hpp"

#include "../../../clock.hpp"
#include "../../pipeline/cache.hpp"
#include "../../pipeline/named.hpp"
#include "../../state.hpp"

#include <string>

namespace rund::node::accel::detail {

#if defined(__APPLE__) && defined(RUND_NODE_HAVE_METAL_SDK)
namespace {

[[nodiscard]] std::string MetalViewSource() {
  return R"MSL(
#include <metal_stdlib>
using namespace metal;

struct ViewParams {
  ulong count;
  ulong offset_words;
  ulong stride_words;
  uint element_words;
  uint reserved;
};

kernel void rund_pipeline_view_gather(
    device const uint *source [[buffer(0)]],
    device uint *dense [[buffer(1)]],
    constant ViewParams &params [[buffer(2)]],
    uint gid [[thread_position_in_grid]]) {
  if (ulong(gid) >= params.count) { return; }
  const ulong source_word =
      params.offset_words + ulong(gid) * params.stride_words;
  const ulong dense_word = ulong(gid) * ulong(params.element_words);
  dense[dense_word] = source[source_word];
  if (params.element_words == 2u) {
    dense[dense_word + 1u] = source[source_word + 1u];
  }
}

kernel void rund_pipeline_view_scatter(
    device const uint *dense [[buffer(0)]],
    device uint *target [[buffer(1)]],
    constant ViewParams &params [[buffer(2)]],
    uint gid [[thread_position_in_grid]]) {
  if (ulong(gid) >= params.count) { return; }
  const ulong dense_word = ulong(gid) * ulong(params.element_words);
  const ulong target_word =
      params.offset_words + ulong(gid) * params.stride_words;
  target[target_word] = dense[dense_word];
  if (params.element_words == 2u) {
    target[target_word + 1u] = dense[dense_word + 1u];
  }
}
)MSL";
}

} // namespace

std::shared_ptr<void> AcquireMetalViewPipeline(MetalAdapter &adapter,
                                               const char *const key,
                                               const char *const function) {
  std::shared_ptr<void> pipeline = LookupMetalNamedPipeline(adapter, key);
  if (pipeline != nullptr) {
    return pipeline;
  }
  std::shared_ptr<void> library =
      AcquireMetalLibrary(adapter, MetalViewSource());
  id<MTLDevice> device = (__bridge id<MTLDevice>)adapter.device.get();
  id<MTLLibrary> native_library = (__bridge id<MTLLibrary>)library.get();
  const std::uint64_t begin = MonotonicNanoseconds();
  if (!MakeNamedMetalPipeline(device, native_library, function, pipeline)) {
    return {};
  }
  StoreMetalNamedPipeline(adapter, key, pipeline,
                          MonotonicNanoseconds() - begin);
  return pipeline;
}

#endif

} // namespace rund::node::accel::detail

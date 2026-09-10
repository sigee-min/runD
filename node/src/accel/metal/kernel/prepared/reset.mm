#include "local.hpp"

#include "../../pipeline/named.hpp"
#include "../../resident.hpp"

#include "../../../kernel/reset/projection.hpp"
#include "../../../kernel/reset/proof.hpp"
#include "../../../kernel/reset/stats.hpp"

#include <accel/device.hpp>

#include <kernel/core/checked.hpp>
#include <rund/counter.hpp>

#include <limits>

namespace rund::node::accel::detail {

#if defined(__APPLE__) && defined(RUND_NODE_HAVE_METAL_SDK)
namespace {

[[nodiscard]] std::string ResetSource() {
  return R"MSL(
#include <metal_stdlib>
using namespace metal;
struct ResetParams {
  ulong count;
  ulong base;
  ulong offset_words;
  ulong stride_words;
  uint element_words;
  uint reserved;
};
kernel void rund_compute_reset(device uint *target [[buffer(0)]],
                               constant ResetParams &params [[buffer(1)]],
                               uint gid [[thread_position_in_grid]]) {
  const ulong ordinal = params.base + ulong(gid);
  if (ordinal >= params.count) { return; }
  const ulong word = params.offset_words + ordinal * params.stride_words;
  target[word] = 0u;
  if (params.element_words == 2u) { target[word + 1u] = 0u; }
}
)MSL";
}

} // namespace

[[nodiscard]] bool PrepareMetalResets(const rund::AccelDevice &pick,
                                      MetalAdapter &adapter,
                                      const BoundResets *const resets,
                                      MetalKernelResources &resources) {
  if (resets == nullptr || resets->size() == 0u) {
    return true;
  }
  resources.resets.reserve(static_cast<std::size_t>(resets->size()));
  for (std::uint64_t index = 0u; index < resets->size(); ++index) {
    const BoundReset &route = (*resets)[static_cast<std::size_t>(index)];
    const rund::kernel::ResidentBufferRef ref = route.ref();
    const MetalViewTransfer *replacement = nullptr;
    if (route.external && !reset::Find(resources, route.binding, replacement)) {
      return false;
    }
    MetalResidentBufferResult resident =
        replacement == nullptr
            ? LookupMetalResidentBuffer(pick, ref, route.handle())
            : replacement->dense;
    if (!resident.check.ok || resident.device_buffer == nullptr) {
      return false;
    }
    const reset::Replacement dense{
        .count = replacement == nullptr ? 0u : replacement->count,
        .element = replacement == nullptr ? 0u : replacement->element_bytes,
    };
    id<MTLBuffer> buffer = (__bridge id<MTLBuffer>)resident.device_buffer.get();
    reset::Range range = route.range();
    if (replacement != nullptr) {
      const reset::Range proved =
          reset::Prove(reset::Project(ref, &dense),
                       static_cast<std::uint64_t>(buffer.length));
      if (!proved.valid()) {
        return false;
      }
      range = proved;
    } else if (!range.valid()) {
      return false;
    }
    resources.resets.push_back(MetalReset{
        .resident = std::move(resident),
        .range = range,
    });
    constexpr std::uint64_t window = std::numeric_limits<std::uint32_t>::max();
    resources.reset_count = ::rund::detail::counter::SaturatingAdd(
        resources.reset_count, reset::Commands(range.count(), window));
    resources.reset_bytes = ::rund::detail::counter::SaturatingAdd(
        resources.reset_bytes, reset::Payload(range));
  }
  resources.reset_pipeline = LookupMetalNamedPipeline(adapter, "compute.reset");
  if (resources.reset_pipeline == nullptr) {
    std::shared_ptr<void> library = AcquireMetalLibrary(adapter, ResetSource());
    id<MTLDevice> device = (__bridge id<MTLDevice>)adapter.device.get();
    id<MTLLibrary> native = (__bridge id<MTLLibrary>)library.get();
    const std::uint64_t begin = MonotonicNanoseconds();
    if (!MakeNamedMetalPipeline(device, native, "rund_compute_reset",
                                resources.reset_pipeline)) {
      return false;
    }
    StoreMetalNamedPipeline(adapter, "compute.reset", resources.reset_pipeline,
                            MonotonicNanoseconds() - begin);
  }
  return resources.reset_pipeline != nullptr;
}

#endif

} // namespace rund::node::accel::detail

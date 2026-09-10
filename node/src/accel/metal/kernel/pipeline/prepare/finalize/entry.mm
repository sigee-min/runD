#include "internal.hpp"

namespace rund::node::accel::detail {

#if defined(__APPLE__) && defined(RUND_NODE_HAVE_METAL_SDK)

rund::AccelCheck MetalPipelineBuild::Finalize(std::shared_ptr<void> &prepared,
                                              PreparedPipelineMemory &memory) {
  if (finished) {
    return rund::AccelCheck{true, "ok"};
  }
  MetalPipelineFinalizeProjection projection{};
  const rund::AccelCheck capture = FinalizeMetalCapture(*this);
  if (!capture.ok) {
    return capture;
  }
  const rund::AccelCheck projected = FinalizeMetalProjection(*this, projection);
  if (!projected.ok) {
    return projected;
  }
  const rund::AccelCheck native = FinalizeMetalNative(*this, projection);
  if (!native.ok) {
    return native;
  }
  return FinalizeMetalOwner(*this, projection, prepared, memory);
}

#endif

} // namespace rund::node::accel::detail

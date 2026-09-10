#include "../internal.hpp"

namespace rund::node::accel::detail {

#if defined(__APPLE__) && defined(RUND_NODE_HAVE_METAL_SDK)

rund::AccelCheck
FinalizeMetalProjection(MetalPipelineBuild &build,
                        MetalPipelineFinalizeProjection &projection) {
  const rund::AccelCheck identity =
      FinalizeMetalProjectionIdentity(build, projection);
  if (!identity.ok) {
    return identity;
  }
  const rund::AccelCheck windows =
      FinalizeMetalProjectionWindows(build, projection);
  if (!windows.ok) {
    return windows;
  }
  return FinalizeMetalProjectionRanges(build, projection);
}

#endif

} // namespace rund::node::accel::detail

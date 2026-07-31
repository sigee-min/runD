#include "local.hpp"

#if !defined(__APPLE__) && !defined(__FreeBSD__) && !defined(__linux__)

namespace rund::node {

BatchIoProbeResult ProbeReactorPlatformNow(
    ReactorPlatform& platform, const BatchIoPollRequest* const requests,
    const std::size_t count, std::vector<BatchIoReady>& out) noexcept {
  return PollPosixReadyNow(requests, count, out,
                           PortableReactorState(platform).probe);
}

}  // namespace rund::node

#endif

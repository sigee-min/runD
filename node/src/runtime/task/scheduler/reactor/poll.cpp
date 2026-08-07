#include "poll.hpp"

#include "../../../reactor/platform.hpp"

namespace rund::node {

ReactorProbeResult ReactorProbeNow(ReactorPlatform &platform,
                                   std::vector<BatchIoReady> &scratch,
                                   const ReactorHandle handle,
                                   const ReactorInterest interest) noexcept {
  const BatchIoPollRequest probe{
      .index = 0u,
      .handle = handle,
      .interest = interest,
  };
  const BatchIoProbeResult result =
      ProbeReactorPlatformNow(platform, &probe, 1u, scratch);
  if (!result.ok) {
    return result.unavailable ? ReactorProbeResult::backend_unavailable()
                              : ReactorProbeResult::poll_failed();
  }
  if (scratch.empty()) {
    return ReactorProbeResult::not_ready();
  }
  const BatchIoReady &ready = scratch.front();
  return ready.invalid ? ReactorProbeResult::invalid(ready.events)
                       : ReactorProbeResult::ready(ready.events);
}

} // namespace rund::node

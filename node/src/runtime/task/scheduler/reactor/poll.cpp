#include "poll.hpp"

#include "../../../reactor/platform.hpp"

namespace rund::node {

ReactorProbeResult ReactorProbeNow(ReactorPlatform &platform,
                                   std::vector<BatchIoReady> &scratch,
                                   const ReactorRequest &request) noexcept {
  const BatchIoPollRequest probe{
      .index = 0u,
      .handle = request.fd,
      .interest = request.interest,
  };
  const BatchIoProbeResult result =
      ProbeReactorPlatformNow(platform, &probe, 1u, scratch);
  if (!result.ok) {
    return ReactorProbeResult{
        .failed = true,
        .unavailable = result.unavailable,
    };
  }
  if (scratch.empty()) {
    return {};
  }
  const BatchIoReady &ready = scratch.front();
  return ReactorProbeResult{
      .ready =
          ReactorReady{
              .wait_id = request.wait_id,
              .task_id = request.task_id,
              .fd = request.fd,
              .interest = request.interest,
              .events = ready.events,
              .invalid = ready.invalid,
          },
      .has_ready = true,
  };
}

} // namespace rund::node

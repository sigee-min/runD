#include "../../reactor/platform.hpp"

namespace rund::node {

struct ReactorPlatformState {};

void ReactorPlatformStateDelete::operator()(
    ReactorPlatformState *const state) const noexcept {
  delete state;
}

ReactorPlatformOpResult
PrepareReactorPlatform(ReactorPlatform &platform,
                       const std::size_t capacity) noexcept {
  (void)platform;
  (void)capacity;
  return {};
}

ReactorPlatformOpResult
OpenReactorPlatform(ReactorPlatform &platform) noexcept {
  (void)platform;
  return ReactorPlatformOpResult{.ok = false, .unavailable = true};
}

void CloseReactorPlatform(ReactorPlatform &platform) noexcept {
  platform.state.reset();
}

ReactorPlatformOpResult
AddReactorPlatformInterest(ReactorPlatform &platform,
                           const ReactorHandle handle,
                           const ReactorInterest interest) noexcept {
  (void)platform;
  (void)handle;
  (void)interest;
  return ReactorPlatformOpResult{.ok = false, .unavailable = true};
}

ReactorPlatformOpResult
ModifyReactorPlatformInterest(ReactorPlatform &platform,
                              const ReactorHandle handle,
                              const ReactorInterest interest) noexcept {
  (void)platform;
  (void)handle;
  (void)interest;
  return ReactorPlatformOpResult{.ok = false, .unavailable = true};
}

ReactorPlatformOpResult
RemoveReactorPlatformInterest(ReactorPlatform &platform,
                              const ReactorHandle handle) noexcept {
  (void)platform;
  (void)handle;
  return ReactorPlatformOpResult{.ok = false, .unavailable = true};
}

ReactorPlatformBatchResult
ApplyReactorPlatformChanges(ReactorPlatform &platform,
                            const ReactorRegistrationChange *const changes,
                            const std::size_t count) noexcept {
  (void)platform;
  (void)changes;
  (void)count;
  return ReactorPlatformBatchResult{.ok = false, .unavailable = true};
}

ReactorPlatformPollResult
PollReactorPlatform(ReactorPlatform &platform, const int timeout_ms,
                    const std::size_t max_events) noexcept {
  (void)platform;
  (void)timeout_ms;
  (void)max_events;
  return ReactorPlatformPollResult{.ok = false, .unavailable = true};
}

BatchIoProbeResult ProbeReactorPlatformNow(
    ReactorPlatform &platform, const BatchIoPollRequest *const requests,
    const std::size_t count, std::vector<BatchIoReady> &out) noexcept {
  (void)platform;
  (void)requests;
  (void)count;
  out.clear();
  return BatchIoProbeResult{.ok = false, .unavailable = true};
}

} // namespace rund::node

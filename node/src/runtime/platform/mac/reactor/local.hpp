#pragma once

#include "../../../reactor/platform.hpp"
#include "../../io.hpp"
#include "../../posix/probe.hpp"

#if defined(__APPLE__) || defined(__FreeBSD__)

#include <sys/event.h>

#include <cstddef>
#include <cstdint>
#include <ctime>
#include <vector>

namespace rund::node {

struct ReactorPlatformState {
  int native = -1;
  bool opened = false;
  std::vector<ReactorPlatformRegistration> registrations{};
  std::vector<ReactorPlatformReady> ready{};
  std::vector<struct kevent> changes{};
  std::vector<struct kevent> receipts{};
  std::vector<struct kevent> events{};
  std::vector<std::size_t> owners{};
  std::vector<bool> ignore_missing{};
  PosixProbeBuffer probe{};
};

[[nodiscard]] inline ReactorPlatformState& MacReactorState(
    ReactorPlatform& platform) noexcept {
  return *platform.state;
}

[[nodiscard]] inline const ReactorPlatformState& MacReactorState(
    const ReactorPlatform& platform) noexcept {
  return *platform.state;
}

[[nodiscard]] timespec KqueueTimeoutSpec(int timeout_ms) noexcept;
void KqueuePushFilter(std::vector<struct kevent>& changes, ReactorHandle handle,
                      short filter, std::uint16_t flags);
void KqueuePushReceiptFilter(std::vector<struct kevent>& changes,
                             std::vector<std::size_t>& owners,
                             std::vector<bool>& ignore_missing,
                             std::size_t owner, bool ignore_enoent,
                             ReactorHandle handle, short filter,
                             std::uint16_t flags);

[[nodiscard]] ReactorPlatformOpResult KqueueSubmit(
    ReactorPlatform& platform, struct kevent* changes, int count,
    bool ignore_missing) noexcept;
[[nodiscard]] ReactorPlatformBatchResult KqueueSubmitBatch(
    ReactorPlatform& handle) noexcept;

[[nodiscard]] bool KqueueReserveInterestStorage(
    ReactorPlatform& platform, ReactorHandle native_handle) noexcept;
[[nodiscard]] bool KqueueRememberInterest(ReactorPlatform& platform,
                                          ReactorHandle native_handle,
                                          ReactorInterest interest) noexcept;
void KqueueForgetInterest(ReactorPlatform& platform,
                          ReactorHandle native_handle) noexcept;
[[nodiscard]] bool KqueueReserveBatchInterestStorage(
    ReactorPlatform& handle, const ReactorRegistrationChange* changes,
    std::size_t count) noexcept;
void KqueueRememberBatchInterest(ReactorPlatform& handle,
                                 const ReactorRegistrationChange* changes,
                                 std::size_t count) noexcept;
[[nodiscard]] ReactorPlatformOpResult KqueueApplyOneChange(
    ReactorPlatform& handle, const ReactorRegistrationChange& change) noexcept;

[[nodiscard]] ReactorPlatformOpResult KqueueAddInterest(
    ReactorPlatform& handle, ReactorHandle native_handle,
    ReactorInterest interest) noexcept;
[[nodiscard]] ReactorPlatformOpResult KqueueRemoveInterest(
    ReactorPlatform& handle, ReactorHandle native_handle,
    bool ignore_missing) noexcept;

[[nodiscard]] ReactorEvent KqueueReadyEvents(const struct kevent& event,
                                             ReactorInterest interest) noexcept;
[[nodiscard]] ReactorInterest KqueueInterestForHandle(
    const ReactorPlatform& platform, ReactorHandle handle) noexcept;
[[nodiscard]] bool KqueueFindInvalidRegistration(
    const ReactorPlatform& handle, ReactorPlatformReady* out) noexcept;

}  // namespace rund::node

#endif

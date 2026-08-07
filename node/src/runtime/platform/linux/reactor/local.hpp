#pragma once

#include "../../../reactor/platform.hpp"
#include "../../io.hpp"
#include "../../posix/probe.hpp"

#if defined(__linux__)

#include <sys/epoll.h>

#include <cstddef>
#include <cstdint>
#include <vector>

namespace rund::node {

struct ReactorPlatformState {
  int native = -1;
  bool opened = false;
  std::vector<ReactorPlatformRegistration> registrations{};
  std::vector<epoll_event> events{};
  PosixProbeBuffer probe{};
};

[[nodiscard]] inline ReactorPlatformState& LinuxReactorState(
    ReactorPlatform& platform) noexcept {
  return *platform.state;
}

[[nodiscard]] inline const ReactorPlatformState& LinuxReactorState(
    const ReactorPlatform& platform) noexcept {
  return *platform.state;
}

[[nodiscard]] std::uint32_t EpollEventsForInterest(
    ReactorInterest interest) noexcept;
[[nodiscard]] ReactorEvent EpollReadyEvents(std::uint32_t events,
                                            ReactorInterest interest) noexcept;
[[nodiscard]] ReactorPlatformReady EpollReadyEvent(
    const ReactorPlatform& handle, const epoll_event& event) noexcept;
[[nodiscard]] ReactorInterest EpollInterestForHandle(
    const ReactorPlatform& platform, ReactorHandle handle) noexcept;
[[nodiscard]] bool EpollFindInvalidRegistration(
    const ReactorPlatform& handle, ReactorPlatformReady* out) noexcept;

[[nodiscard]] bool EpollReserveInterestStorage(
    ReactorPlatform& handle, ReactorHandle native_handle) noexcept;
[[nodiscard]] bool EpollRememberInterest(ReactorPlatform& handle,
                                         ReactorHandle native_handle,
                                         ReactorInterest interest) noexcept;
void EpollForgetInterest(ReactorPlatform& handle,
                         ReactorHandle native_handle) noexcept;
[[nodiscard]] ReactorPlatformOpResult EpollCtl(
    ReactorPlatform& handle, int op, ReactorHandle native_handle,
    ReactorInterest interest) noexcept;

}  // namespace rund::node

#endif

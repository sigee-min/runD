#pragma once

#ifndef _XOPEN_SOURCE
#define _XOPEN_SOURCE 700
#endif

#include "../../../reactor/platform.hpp"
#include "../../io.hpp"
#include "../../posix/probe.hpp"

#if !defined(__APPLE__) && !defined(__FreeBSD__) && !defined(__linux__)

#include <poll.h>

#include <cstddef>
#include <vector>

namespace rund::node {

struct ReactorPlatformState {
  bool opened = false;
  std::vector<ReactorPlatformRegistration> registrations{};
  std::vector<ReactorPlatformReady> ready{};
  std::vector<pollfd> events{};
  PosixProbeBuffer probe{};
};

[[nodiscard]] inline ReactorPlatformState& PortableReactorState(
    ReactorPlatform& platform) noexcept {
  return *platform.state;
}

[[nodiscard]] inline const ReactorPlatformState& PortableReactorState(
    const ReactorPlatform& platform) noexcept {
  return *platform.state;
}

[[nodiscard]] bool PollReserveInterestStorage(
    ReactorPlatform& handle, ReactorHandle native_handle) noexcept;
[[nodiscard]] bool PollRememberInterest(ReactorPlatform& handle,
                                        ReactorHandle native_handle,
                                        ReactorInterest interest) noexcept;
void PollForgetInterest(ReactorPlatform& handle,
                        ReactorHandle native_handle) noexcept;
[[nodiscard]] bool PollReadyEvent(const pollfd& descriptor,
                                  ReactorPlatformReady* out) noexcept;

}  // namespace rund::node

#endif

#pragma once

#include <poll.h>

#include <cstddef>
#include <vector>

#include "../../reactor/platform.hpp"

namespace rund::node {

struct PosixProbeBuffer {
  std::vector<pollfd> events{};
};

[[nodiscard]] int PosixFd(ReactorHandle handle) noexcept;
[[nodiscard]] short PosixInterest(ReactorInterest interest) noexcept;
[[nodiscard]] ReactorEvent PosixEvents(short events) noexcept;
[[nodiscard]] bool IoPollInvalid(short revents) noexcept;
[[nodiscard]] BatchIoProbeResult PollPosixReadyNow(
    const BatchIoPollRequest* requests, std::size_t count,
    std::vector<BatchIoReady>& out, PosixProbeBuffer& buffer) noexcept;

}  // namespace rund::node

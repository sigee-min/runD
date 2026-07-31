#pragma once

#include <cstdint>

namespace rund::host {

enum class Status : std::uint16_t {
  Ok = 0u,
  Invalid = 1u,
  CapacityExceeded = 2u,
  SyscallFailed = 3u,
  ReplayMismatch = 4u,
  WouldBlock = 5u,
  Unsupported = 6u,
};

} // namespace rund::host

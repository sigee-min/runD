#pragma once

#include <cstdint>

namespace rund {

enum class SessionState : std::uint8_t {
  Unconfigured,
  Configured,
  Running,
  Draining,
  Stopped,
};

} // namespace rund

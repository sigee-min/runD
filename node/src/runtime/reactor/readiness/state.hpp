#pragma once

// Width-safe scheduler readiness state. Native platform records and native
// readiness masks never cross this boundary.

#include <cstdint>
#include <limits>

namespace rund::node {

using ReactorHandle = std::uintptr_t;

inline constexpr ReactorHandle kInvalidReactorHandle =
    std::numeric_limits<ReactorHandle>::max();

enum class ReactorInterest : std::uint8_t {
  None = 0u,
  Read = 1u,
  Write = 4u,
};

enum class ReactorEvent : std::uint8_t {
  None = 0u,
  Read = 1u,
  Write = 4u,
  Error = 8u,
  Hangup = 16u,
};

} // namespace rund::node

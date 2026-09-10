#pragma once

#include "invocation.hpp"

namespace rund::compute::detail::residency::execution {

enum class SlidingTicketKind : std::uint8_t {
  Fetch,
  Promote,
  Native,
  Drain,
  Persist,
};

// Capability binds plan, coordinate, Authority token, run generation, owner
// nonce, cyclic turn, byte extent, slot, use, and transition kind.
struct SlidingTicket final {
  residency::Identity plan{};
  SlidingCoordinate coordinate{};
  std::uint64_t token{};
  std::uint64_t generation{};
  std::uint64_t owner{};
  std::uint64_t turn{};
  std::uint64_t expected_bytes{};
  std::uint32_t slot{};
  std::uint32_t use{};
  SlidingTicketKind kind{SlidingTicketKind::Fetch};
};

} // namespace rund::compute::detail::residency::execution

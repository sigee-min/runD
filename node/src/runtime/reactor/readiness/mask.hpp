#pragma once

#include "state.hpp"

#include <cstdint>

namespace rund::node {

[[nodiscard]] constexpr ReactorInterest operator|(const ReactorInterest left,
                                                  const ReactorInterest right) {
  return static_cast<ReactorInterest>(static_cast<std::uint8_t>(left) |
                                      static_cast<std::uint8_t>(right));
}

[[nodiscard]] constexpr bool
HasReactorInterest(const ReactorInterest value,
                   const ReactorInterest flag) noexcept {
  return (static_cast<std::uint8_t>(value) & static_cast<std::uint8_t>(flag)) !=
         0u;
}

[[nodiscard]] constexpr ReactorEvent operator|(const ReactorEvent left,
                                               const ReactorEvent right) {
  return static_cast<ReactorEvent>(static_cast<std::uint8_t>(left) |
                                   static_cast<std::uint8_t>(right));
}

constexpr ReactorEvent &operator|=(ReactorEvent &left,
                                   const ReactorEvent right) noexcept {
  left = left | right;
  return left;
}

[[nodiscard]] constexpr bool HasReactorEvent(const ReactorEvent value,
                                             const ReactorEvent flag) noexcept {
  return (static_cast<std::uint8_t>(value) & static_cast<std::uint8_t>(flag)) !=
         0u;
}

[[nodiscard]] constexpr short
ReactorInterestBits(const ReactorInterest interest) noexcept {
  return static_cast<short>(interest);
}

[[nodiscard]] constexpr short
ReactorEventBits(const ReactorEvent events) noexcept {
  return static_cast<short>(events);
}

[[nodiscard]] constexpr ReactorInterest
ReactorInterestFromBits(const short bits) noexcept {
  return static_cast<ReactorInterest>(static_cast<std::uint8_t>(bits));
}

[[nodiscard]] constexpr ReactorEvent
ReactorEventsForInterest(const ReactorInterest interest) noexcept {
  ReactorEvent events = ReactorEvent::None;
  if (HasReactorInterest(interest, ReactorInterest::Read)) {
    events |= ReactorEvent::Read;
  }
  if (HasReactorInterest(interest, ReactorInterest::Write)) {
    events |= ReactorEvent::Write;
  }
  return events;
}

[[nodiscard]] constexpr bool
ReactorEventsMatch(const ReactorEvent events,
                   const ReactorInterest interest) noexcept {
  return HasReactorEvent(events, ReactorEvent::Error) ||
         HasReactorEvent(events, ReactorEvent::Hangup) ||
         (HasReactorEvent(events, ReactorEvent::Read) &&
          HasReactorInterest(interest, ReactorInterest::Read)) ||
         (HasReactorEvent(events, ReactorEvent::Write) &&
          HasReactorInterest(interest, ReactorInterest::Write));
}

} // namespace rund::node

#pragma once

#include <cstdint>
#include <optional>
#include <rund/net/ready/set.hpp>
#include <rund/net/socket.hpp>
#include <rund/reason.hpp>

#include "../../../reactor/readiness/state.hpp"

namespace rund::node {

struct ReactorManyRequest {
  std::uint64_t group_id = 0u;
  std::uint64_t wait_id = 0u;
  ::rund::net::SocketView socket{};
  ReactorHandle fd = kInvalidReactorHandle;
  std::uint32_t slot = 0u;
  std::uint32_t event_index = 0u;
  ReactorInterest interest = ReactorInterest::None;
};

struct ReactorManyEvent {
  std::uint64_t group_id = 0u;
  ::rund::net::SocketView socket{};
  ReactorHandle fd = kInvalidReactorHandle;
  std::uint32_t slot = 0u;
  std::uint32_t event_index = 0u;
  ReactorInterest interest = ReactorInterest::None;
  ReactorEvent events = ReactorEvent::None;
  ReasonCode code = ReasonCode::Ok;
};

using ReactorManyEventSlot = std::optional<ReactorManyEvent>;

struct ReactorManyGroup {
  std::uint64_t group_id = 0u;
  std::uint64_t task_id = 0u;
  ::rund::net::ready::Set ready_set{};
  std::uint64_t timer_wait_id = 0u;
  ::rund::detail::task::StopSourceIdentity stop{};
  std::uint32_t first_request = 0u;
  std::uint32_t request_count = 0u;
  std::uint32_t max_events = 0u;
  std::uint32_t stored_event_count = 0u;
  bool completed = false;
  bool budget_exhausted = false;
};

} // namespace rund::node

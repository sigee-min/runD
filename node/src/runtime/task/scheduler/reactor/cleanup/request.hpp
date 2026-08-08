#pragma once

#include <rund/task/handle.hpp>
#include <rund/reason.hpp>

#include <cstdint>
#include "../many.hpp"
#include "../model.hpp"

namespace rund::node {

class Scheduler;

enum class ReactorTimeoutCleanupPolicy : std::uint8_t {
  None,
  IfPresent,
  Required,
};

struct ReactorCleanupRequest {
  std::uint64_t wait_id = 0u;
  std::uint64_t group_id = 0u;
  ReasonCode reason = ReasonCode::Ok;
  ReactorTimeoutCleanupPolicy timeout_cleanup =
      ReactorTimeoutCleanupPolicy::None;
  bool remove_ready_backlog = false;
  bool cleanup_siblings = false;
  bool erase_group = false;
  bool wake_owner = true;
  ReactorEvent events = ReactorEvent::None;
  bool store_event = false;
  std::int64_t deadline_ns = 0;
};

struct ReactorRemovedWaitCleanupRequest {
  ReactorWait wait{};
  ReasonCode reason = ReasonCode::Ok;
  ReactorTimeoutCleanupPolicy timeout_cleanup =
      ReactorTimeoutCleanupPolicy::None;
  bool remove_ready_backlog = false;
  bool cleanup_siblings = false;
  bool wake_owner = true;
  ReactorEvent events = ReactorEvent::None;
  bool store_event = false;
  std::int64_t deadline_ns = 0;
};

[[nodiscard]] bool ReactorCleanupWait(
    Scheduler& scheduler,
    ReactorCleanupRequest request) noexcept;

[[nodiscard]] bool ReactorCleanupRemovedWait(
    Scheduler& scheduler,
    ReactorRemovedWaitCleanupRequest request) noexcept;

}  // namespace rund::node

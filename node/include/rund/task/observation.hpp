#pragma once

#include <rund/reason.hpp>

#include <cstdint>

namespace rund::task {

enum class ObservationKind : std::uint16_t {
  None = 0u,
  TimerDue = 1u,
  IoReady = 2u,
  IoInvalid = 3u,
  IoPollFailed = 4u,
};

struct Observation {
  std::uint64_t sequence = 0u;
  ObservationKind kind = ObservationKind::None;
  std::uint64_t task_id = 0u;
  std::uint64_t wait_id = 0u;
  int fd = -1;
  short interest = 0;
  short revents = 0;
  std::int64_t deadline_ns = 0;
  ReasonCode reason_code = ReasonCode::Ok;
};

} // namespace rund::task

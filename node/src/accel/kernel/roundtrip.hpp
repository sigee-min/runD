#pragma once

#include <accel/kernel/run.hpp>

#include "schedule.hpp"
#include <node/accel/context.hpp>

#include <cstdint>

namespace rund::node::accel::detail {

struct ProducerConsumerRoundtrip {
  std::uint64_t internal_bytes = 0u;
  std::uint64_t external_bytes = 0u;
  bool ok = true;
  const char *reason = "ok";
};

[[nodiscard]] ProducerConsumerRoundtrip
CountProducerConsumerRoundtripBytes(const KernelExecution &execution,
                                    const ScheduledStepOrder &step_order,
                                    const rund::AccelRun &run) noexcept;

} // namespace rund::node::accel::detail

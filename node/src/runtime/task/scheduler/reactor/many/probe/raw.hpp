#pragma once

#include <rund/task/handle.hpp>
#include <span>
#include <vector>

#include "../../../../../reactor/platform.hpp"
#include "../../many.hpp"

namespace rund::node {

class Scheduler;
struct ReactorPlatform;

struct ReactorManyProbeResult {
  ReasonCode code = ReasonCode::Ok;
  std::uint32_t total_ready = 0u;

  [[nodiscard]] constexpr bool ok() const noexcept {
    return code == ReasonCode::Ok;
  }
};

struct ReactorManyProbeBatchResult {
  bool ok = true;
  bool unavailable = false;
};

[[nodiscard]] ReactorManyProbeResult ReactorProbeManyReady(
    Scheduler &scheduler, ReactorPlatform &platform, std::uint64_t task_id,
    std::uint32_t limit, std::span<const ReactorManyRequest> requests,
    std::vector<BatchIoPollRequest> &poll_requests,
    std::vector<BatchIoReady> &ready_results, ReactorManyGroup &group,
    std::vector<ReactorManyEventSlot> &event_slots) noexcept;

[[nodiscard]] ReactorManyProbeBatchResult
ReactorProbeManyReadyNow(ReactorPlatform &platform,
                         std::span<const ReactorManyRequest> requests,
                         std::vector<BatchIoPollRequest> &poll_requests,
                         std::vector<BatchIoReady> &ready) noexcept;

[[nodiscard]] const ReactorManyRequest *
ReactorManyProbeRequestForReady(std::span<const ReactorManyRequest> requests,
                                const BatchIoReady &ready) noexcept;

} // namespace rund::node

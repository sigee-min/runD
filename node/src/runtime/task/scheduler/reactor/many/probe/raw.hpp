#pragma once

#include <limits>
#include <rund/task/handle.hpp>
#include <span>
#include <vector>

#include "../../../../../reactor/platform.hpp"
#include "../../many.hpp"

namespace rund::node {

class Scheduler;
struct ReactorPlatform;

class ReactorManyProbeResult final {
public:
  [[nodiscard]] static constexpr ReactorManyProbeResult
  success(const std::uint32_t total_ready = 0u) noexcept {
    return ReactorManyProbeResult{ReasonCode::Ok, total_ready};
  }

  [[nodiscard]] static constexpr ReactorManyProbeResult
  wait_capacity_exceeded() noexcept {
    return ReactorManyProbeResult{ReasonCode::ReactorWaitCapacityExceeded, 0u};
  }

  [[nodiscard]] static constexpr ReactorManyProbeResult
  backend_unavailable() noexcept {
    return ReactorManyProbeResult{ReasonCode::ReactorBackendUnavailable, 0u};
  }

  [[nodiscard]] static constexpr ReactorManyProbeResult poll_failed() noexcept {
    return ReactorManyProbeResult{ReasonCode::IoPollFailed, 0u};
  }

  [[nodiscard]] static constexpr ReactorManyProbeResult
  host_replay_mismatch(const std::uint32_t ready_prefix = 0u) noexcept {
    return ReactorManyProbeResult{ReasonCode::HostReplayEventMismatch,
                                  ready_prefix};
  }

  [[nodiscard]] static constexpr ReactorManyProbeResult
  invalid_after(const std::uint32_t ready_prefix) noexcept {
    return ready_prefix == std::numeric_limits<std::uint32_t>::max()
               ? wait_capacity_exceeded()
               : ReactorManyProbeResult{ReasonCode::IoFdInvalid,
                                        ready_prefix + 1u};
  }

  [[nodiscard]] constexpr bool ok() const noexcept {
    return code_ == ReasonCode::Ok;
  }

  [[nodiscard]] constexpr ReasonCode code() const noexcept { return code_; }

  [[nodiscard]] constexpr std::uint32_t total_ready() const noexcept {
    return total_ready_;
  }

private:
  constexpr ReactorManyProbeResult(const ReasonCode code,
                                   const std::uint32_t total_ready) noexcept
      : code_(code), total_ready_(total_ready) {}

  ReasonCode code_;
  std::uint32_t total_ready_;
};

[[nodiscard]] ReactorManyProbeResult ReactorProbeManyReady(
    Scheduler &scheduler, ReactorPlatform &platform, std::uint64_t task_id,
    std::uint32_t limit, std::span<const ReactorManyRequest> requests,
    std::vector<BatchIoPollRequest> &poll_requests,
    std::vector<BatchIoReady> &ready_results, ReactorManyGroup &group,
    std::vector<ReactorManyEventSlot> &event_slots) noexcept;

[[nodiscard]] BatchIoProbeResult
ReactorProbeManyReadyNow(ReactorPlatform &platform,
                         std::span<const ReactorManyRequest> requests,
                         std::vector<BatchIoPollRequest> &poll_requests,
                         std::vector<BatchIoReady> &ready) noexcept;

[[nodiscard]] const ReactorManyRequest *
ReactorManyProbeRequestForReady(std::span<const ReactorManyRequest> requests,
                                const BatchIoReady &ready) noexcept;

} // namespace rund::node

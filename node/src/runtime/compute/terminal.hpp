#pragma once

#include <rund/compute/status.hpp>

#include <atomic>
#include <cstdint>

namespace rund::node::compute_detail {

struct TaskState;

enum class TerminalPhase : std::uint8_t {
  Open,
  Cancelled,
  Finishing,
  Complete,
};

enum class FinishClaim : std::uint8_t {
  Finish,
  Cancel,
  Closed,
};

enum class CancelClaim : std::uint8_t {
  Accept,
  Cancelled,
  Closed,
  Invalid,
};

[[nodiscard]] FinishClaim ClaimFinish(
    std::atomic<TerminalPhase>& phase) noexcept;
[[nodiscard]] CancelClaim RequestCancel(
    std::atomic<TerminalPhase>& phase) noexcept;
void MarkComplete(std::atomic<TerminalPhase>& phase) noexcept;

[[nodiscard]] compute::Status FinishCpu(TaskState &task) noexcept;
[[nodiscard]] compute::Status FinishAccel(TaskState &task) noexcept;
[[nodiscard]] compute::Status FinishFailure(TaskState &task,
                                            compute::Status status) noexcept;

} // namespace rund::node::compute_detail

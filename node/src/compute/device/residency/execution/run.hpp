#pragma once

#include "../registry/credentials/execution.hpp"
#include "plan.hpp"

#include <array>
#include <cstddef>
#include <cstdint>

namespace rund::compute::detail::residency::execution {

// Q=1 run-owned join for the three independent terminal owners. Native owns
// only Dispatch evidence. Authority owns exact Input/Output service tickets
// and the final combined close; this class cannot publish at native callback.
class Run final {
public:
  Run() = default;
  Run(const Run &) = delete;
  Run &operator=(const Run &) = delete;

  [[nodiscard]] ExecutionLease begin(Authority &, const Plan &) noexcept;
  [[nodiscard]] bool issue(Phase, ExecutionTicket &) noexcept;
  [[nodiscard]] bool terminal(const ExecutionTicket &, ExecutionTerminal,
                              Status) noexcept;
  [[nodiscard]] bool native(const NativeEvidence &) noexcept;
  // Closes the native join when backend submission was synchronously rejected.
  // This is not NativeEvidence: no OS command or callback existed.
  [[nodiscard]] bool reject(Status failure) noexcept;
  // Releases an internally contradictory Q=1 join only after the coordinator
  // has collected every accepted native handoff's final callback. Authority
  // conservatively invalidates the entire reserved journal.
  [[nodiscard]] bool abandon() noexcept;
  [[nodiscard]] ExecutionClose close(std::uint64_t completed_ns) noexcept;

  [[nodiscard]] const ExecutionLease &lease() const noexcept { return lease_; }

private:
  [[nodiscard]] bool merge(FailureEvidence) noexcept;

  Authority *authority_{};
  const Plan *plan_{};
  ExecutionLease lease_{};
  NativeEvidence native_{};
  std::array<FailureEvidence, FailureCapacity> host_failures_{};
  ExecutionTicket input_{};
  ExecutionTicket output_{};
  Status host_status_{Status::success()};
  std::size_t host_failure_count_{};
  std::uint64_t sequence_{1u};
  bool input_issued_{};
  bool input_done_{};
  bool native_done_{};
  bool native_rejected_{};
  bool output_issued_{};
  bool output_done_{};
  bool host_unknown_{};
  bool closed_{};
};

} // namespace rund::compute::detail::residency::execution

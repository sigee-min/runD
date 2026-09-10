#pragma once

#include "service.hpp"
#include "window.hpp"

#include <array>
#include <cstddef>
#include <cstdint>

namespace rund::compute::detail::residency::execution {

// Source-private whole-run facts. Raw chunk Finals are internal and therefore
// do not inflate the one public controller handoff.
struct StreamEvidence final {
  Status status{Status::fail(Reason::PipelineInvalid)};
  TerminalKind terminal{TerminalKind::Known};
  std::uint64_t plan_identity{};
  std::uint64_t token{};
  std::uint64_t generation{};
  std::uint64_t epoch_count{};
  std::uint64_t public_handoffs{};
  std::uint64_t chunk_count{};
  std::uint64_t native_batches{};
  std::uint64_t queue_calls{};
  std::uint64_t native_inflight_peak{};
  std::uint64_t released_prefix{};
  std::uint64_t completed_ns{};
};

// O(1) recurrent authority/controller model for arbitrary Q. Backends still
// submit bounded W<=4 windows; every fixed slot is authenticated by its exact
// run-global epoch before reuse.
class Stream final {
public:
  [[nodiscard]] ExecutionLease begin(Authority &, const Plan &) noexcept;
  // Opens the same fixed Host-service journal under the stronger law that the
  // backend accepts every native batch in one upfront schedule.
  [[nodiscard]] ExecutionLease begin_schedule(Authority &,
                                              const Plan &) noexcept;
  [[nodiscard]] bool issue(std::uint64_t epoch, Phase,
                           ExecutionTicket &) noexcept;
  [[nodiscard]] bool terminal(const ExecutionTicket &, ExecutionTerminal,
                              Status) noexcept;
  [[nodiscard]] bool release(const Release &) noexcept;
  [[nodiscard]] bool chunk(const WindowEvidence &) noexcept;
  [[nodiscard]] bool schedule(const ScheduleEvidence &) noexcept;
  [[nodiscard]] bool abort(Status failure, std::uint64_t completed_ns) noexcept;
  [[nodiscard]] bool abandon_final(const WindowEvidence &) noexcept;
  [[nodiscard]] bool final(StreamEvidence &) const noexcept;
  [[nodiscard]] ExecutionClose close() noexcept;
  [[nodiscard]] bool abandon() noexcept;

  [[nodiscard]] const ExecutionLease &lease() const noexcept { return lease_; }

private:
  struct Slot final {
    Release release{};
    std::uint64_t epoch{NeverUse};
    bool input_issued{};
    bool input_done{};
    bool output_issued{};
    bool output_done{};
  };

  Authority *authority_{};
  const Plan *plan_{};
  ExecutionLease lease_{};
  Service service_{};
  std::array<Slot, WindowCapacity> slots_{};
  Status status_{Status::success()};
  TerminalKind terminal_{TerminalKind::Known};
  std::uint64_t release_count_{};
  std::uint64_t accepted_count_{};
  std::uint64_t chunk_count_{};
  std::uint64_t native_batches_{};
  std::uint64_t queue_calls_{};
  std::uint64_t native_inflight_peak_{};
  std::uint64_t completed_ns_{};
  ExecutionClose closed_result_{};
  bool final_{};
  bool closed_{};
  bool scheduled_{};
};

} // namespace rund::compute::detail::residency::execution

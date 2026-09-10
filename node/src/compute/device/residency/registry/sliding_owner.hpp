#pragma once

#include "../registry.hpp"
#include "../execution/sliding.hpp"

#include <cstddef>
#include <cstdint>
#include <span>

namespace rund::compute::detail::residency {

// Stateless direct owner for the complete Sliding physical/protocol
// lifecycle. Authority remains the sole owner of the gate, frame table,
// execution slot, and all retained credentials; this facet only borrows that
// state and owns the algorithms which authenticate and transition it.
class SlidingOwner final {
public:
  explicit SlidingOwner(Authority &) noexcept;

  SlidingOwner(const SlidingOwner &) noexcept = default;
  SlidingOwner &operator=(const SlidingOwner &) = delete;

  [[nodiscard]] ExecutionLease
  begin_execution_sliding(const execution::Plan &) noexcept;
  [[nodiscard]] bool bind_execution_sliding(execution::Sliding &) noexcept;
  [[nodiscard]] bool abandon_bound_execution_sliding(
      const execution::Plan &, execution::Sliding &, const ExecutionLease &)
      noexcept;
  [[nodiscard]] bool quarantine_bound_execution_sliding(
      const execution::Plan &, execution::Sliding &, Status) noexcept;
  [[nodiscard]] bool seal_execution_sliding_lease(ExecutionLease &) noexcept;
  [[nodiscard]] bool validate_execution_sliding_lease(
      const execution::Plan &, const ExecutionLease &) noexcept;

  [[nodiscard]] bool issue_execution_sliding_fetch(
      const execution::Plan &, execution::Sliding &,
      const execution::SlidingProjection &, std::span<PageUse>, std::size_t,
      execution::SlidingFetch &) noexcept;
  [[nodiscard]] bool terminal_execution_sliding_fetch(
      execution::Sliding &, const execution::SlidingFetch &, Status,
      execution::TerminalKind, bool, std::uint64_t) noexcept;
  [[nodiscard]] bool release_execution_sliding_fetch(
      execution::Sliding &, execution::SlidingFetch &&) noexcept;

  [[nodiscard]] bool issue_execution_sliding_promote(
      const execution::Plan &, execution::Sliding &,
      const execution::SlidingProjection &, std::span<PageUse>,
      execution::SlidingPromote &) noexcept;
  [[nodiscard]] bool terminal_execution_sliding_promote(
      execution::Sliding &, const execution::SlidingPromote &, Status,
      execution::TerminalKind, bool, std::uint64_t) noexcept;
  [[nodiscard]] bool release_execution_sliding_promote(
      execution::Sliding &, execution::SlidingPromote &&) noexcept;

  [[nodiscard]] bool issue_execution_sliding_native(
      const execution::Plan &, execution::Sliding &,
      const execution::SlidingProjection &, std::span<PageUse>,
      execution::SlidingNative &) noexcept;
  [[nodiscard]] bool terminal_execution_sliding_native(
      execution::Sliding &, const execution::SlidingNative &, Status,
      execution::TerminalKind, bool) noexcept;
  [[nodiscard]] bool release_execution_sliding_native(
      execution::Sliding &, execution::SlidingNative &&) noexcept;

  [[nodiscard]] bool issue_execution_sliding_drain(
      const execution::Plan &, execution::Sliding &,
      const execution::SlidingProjection &, std::span<PageUse>, std::size_t,
      execution::SlidingDrain &) noexcept;
  [[nodiscard]] bool terminal_execution_sliding_drain(
      execution::Sliding &, const execution::SlidingDrain &, Status,
      execution::TerminalKind, bool, std::uint64_t) noexcept;
  [[nodiscard]] bool release_execution_sliding_drain(
      execution::Sliding &, execution::SlidingDrain &&) noexcept;

  [[nodiscard]] bool issue_execution_sliding_persist(
      const execution::Plan &, execution::Sliding &,
      const execution::SlidingProjection &, std::span<PageUse>, std::size_t,
      execution::SlidingPersist &) noexcept;
  [[nodiscard]] bool terminal_execution_sliding_persist(
      execution::Sliding &, const execution::SlidingPersist &, Status,
      execution::TerminalKind, bool, std::uint64_t) noexcept;
  [[nodiscard]] bool release_execution_sliding_persist(
      execution::Sliding &, execution::SlidingPersist &&) noexcept;

  [[nodiscard]] bool reject_execution_sliding(const execution::Plan &,
                                              execution::Sliding &, Status)
      noexcept;
  [[nodiscard]] bool reject_execution_sliding_coordinate(
      const execution::Plan &, execution::Sliding &, std::uint64_t, Status)
      noexcept;
  [[nodiscard]] bool quarantine_execution_sliding_coordinate(
      const execution::Plan &, execution::Sliding &, std::uint64_t, Status)
      noexcept;
  [[nodiscard]] bool abandon_execution_sliding(
      const execution::Plan &, const ExecutionLease &) noexcept;

  [[nodiscard]] bool prepare_execution_sliding(
      const execution::Plan &, const execution::SlidingFinal &,
      ExecutionSlidingFinal &) noexcept;
  [[nodiscard]] bool prepare_execution_sliding(
      const execution::Plan &, execution::Sliding &,
      const execution::SlidingFinal &, ExecutionSlidingFinal &,
      FinalAbort * = nullptr) noexcept;
  [[nodiscard]] FinalAbort abort_execution_sliding(
      const execution::Plan &, execution::Sliding &,
      const execution::SlidingFinal &, ExecutionSlidingFinal *, Status) noexcept;
  [[nodiscard]] FinalAbort close_execution_sliding(
      const execution::Plan &, execution::Sliding &, Status) noexcept;
  [[nodiscard]] ExecutionClose accept_execution_sliding(
      ExecutionSlidingFinal &&, ExecutionSlidingReceipt &) noexcept;
  [[nodiscard]] ExecutionClose accept_execution_sliding(
      const execution::Plan &, execution::Sliding &,
      const execution::SlidingFinal &, ExecutionSlidingFinal &&,
      execution::SlidingEvidence &) noexcept;
  [[nodiscard]] ExecutionClose accept_execution_sliding_final(
      const execution::Plan &, execution::Sliding &,
      const execution::SlidingFinal &, ExecutionSlidingFinal &&,
      execution::SlidingEvidence &, void *, ExecutionSlidingPublication)
      noexcept;

private:
  void clear_sliding_frame_observation(std::uint32_t) noexcept;
  void observe_sliding_frame(std::uint32_t, CacheKey) noexcept;
  void quarantine_sliding_locked(execution::Sliding &, Status) noexcept;
  [[nodiscard]] bool close_sliding_locked(execution::Sliding &, Status) noexcept;
  [[nodiscard]] bool commit_sliding_frames(const execution::Plan *, bool)
      noexcept;
  [[nodiscard]] bool sliding_final_matches(
      const execution::Plan &, const execution::Sliding &,
      const execution::SlidingFinal &, const ExecutionSlidingFinal &,
      SlidingFinalPhase) const noexcept;
  [[nodiscard]] ExecutionClose commit_sliding_final(
      const execution::Plan &, execution::Sliding &,
      const execution::SlidingFinal &, ExecutionSlidingFinal &&,
      execution::SlidingEvidence &, void *, ExecutionSlidingPublication)
      noexcept;

  Authority &authority_;
};

} // namespace rund::compute::detail::residency

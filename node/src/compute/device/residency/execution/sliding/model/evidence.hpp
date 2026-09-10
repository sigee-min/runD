#pragma once

#include "ticket.hpp"

namespace rund::compute::detail::residency::execution {

struct SlidingEvidence final {
  Status status{Status::fail(Reason::PipelineInvalid)};
  TerminalKind first_failure_terminal{TerminalKind::Known};
  TerminalKind terminal{TerminalKind::Known};
  residency::Identity plan{};
  SlidingCoordinate first_failure{};
  std::uint64_t token{};
  std::uint64_t generation{};
  std::uint64_t owner{};
  std::uint64_t planned{};
  std::uint64_t admitted{};
  std::uint64_t terminal_frontier{};
  std::uint64_t first_unsent{};
  std::uint64_t fetch_calls{};
  std::uint64_t fetch_hits{};
  std::uint64_t promote_calls{};
  std::uint64_t drain_calls{};
  std::uint64_t persist_calls{};
  std::uint64_t persist_issued{};
  std::uint64_t persist_frontier{};
  std::uint64_t persist_completed_after_failure{};
  std::uint64_t fetch_bytes{};
  std::uint64_t promote_bytes{};
  std::uint64_t drain_bytes{};
  std::uint64_t persist_bytes{};
  std::uint64_t persist_bytes_after_failure{};
  bool has_failure{};
  bool first_failure_may_write{};
  bool quarantined{};
};

class SlidingFinal final {
public:
  SlidingFinal() = default;
  [[nodiscard]] explicit operator bool() const noexcept { return nonce_ != 0u; }
  [[nodiscard]] const SlidingEvidence &evidence() const noexcept {
    return evidence_;
  }

private:
  friend class Sliding;
  friend class ::rund::compute::detail::residency::Authority;
  friend class ::rund::compute::detail::residency::SlidingOwner;
  SlidingEvidence evidence_{};
  std::uint64_t nonce_{};
};

} // namespace rund::compute::detail::residency::execution

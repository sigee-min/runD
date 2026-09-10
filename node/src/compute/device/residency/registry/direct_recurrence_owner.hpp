#pragma once

#include "../registry.hpp"

#include <memory>

namespace rund::compute::detail::residency {

// Stateless owner for the complete service-free Direct recurrence protocol.
// Authority remains the sole owner of the gate, frame table, and execution
// slot; this facet borrows that storage and owns every Direct admission,
// registration, terminal, release, and abort transition.
class DirectRecurrenceOwner final {
public:
  explicit DirectRecurrenceOwner(Authority &) noexcept;

  DirectRecurrenceOwner(const DirectRecurrenceOwner &) noexcept = default;
  DirectRecurrenceOwner &operator=(const DirectRecurrenceOwner &) = delete;

  [[nodiscard]] DirectRecurrenceLease
  begin_direct_recurrence(const DirectRecurrenceRequest &) noexcept;
  [[nodiscard]] bool bind_direct_registration(
      const std::shared_ptr<const registration_detail::State> &) noexcept;
  [[nodiscard]] RegistrationResult retire_direct_registration(
      const std::shared_ptr<const registration_detail::State> &) noexcept;

  [[nodiscard]] bool
  prepare_direct_recurrence_final(const DirectRecurrenceLease &, Status,
                                  execution::TerminalKind, bool may_write,
                                  std::uint64_t completed,
                                  DirectRecurrenceFinal &) noexcept;
  [[nodiscard]] ExecutionClose
  stage_direct_recurrence_final(DirectRecurrenceFinal &&) noexcept;
  [[nodiscard]] RegistrationResult release_direct_recurrence_pending(
      const DirectRecurrenceLease &,
      const std::shared_ptr<const registration_detail::State> &) noexcept;
  [[nodiscard]] ExecutionClose finish_direct_recurrence(
      const DirectRecurrenceLease &,
      const std::shared_ptr<const registration_detail::State> &, void *,
      DirectRecurrencePublication) noexcept;

  [[nodiscard]] DirectAbort
  abort_direct_recurrence(const DirectRecurrenceLease &, Status,
                          execution::TerminalKind, bool may_write) noexcept;
  [[nodiscard]] DirectAbort
  abort_direct_recurrence_frozen(DirectRecurrenceFinal &&) noexcept;

private:
  void quarantine_direct_recurrence() noexcept;
  [[nodiscard]] bool commit_direct_frames(bool success,
                                          bool may_write) noexcept;
  [[nodiscard]] bool
  direct_lease_matches(const DirectRecurrenceLease &) const noexcept;
  [[nodiscard]] bool clean_direct_registration_rows() const noexcept;

  Authority &authority_;
};

} // namespace rund::compute::detail::residency

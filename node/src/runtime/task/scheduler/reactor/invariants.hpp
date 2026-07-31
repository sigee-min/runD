#pragma once

#include <cstdint>

namespace rund::node {

struct ReactorInvariantSnapshot {
  bool ok = false;
  const char *reason = "no_scheduler";
  std::uint64_t waits = 0u;
  std::uint64_t timeout_timers = 0u;
  std::uint64_t ready_backlog_entries = 0u;
  std::uint64_t many_groups = 0u;
  std::uint64_t ready_set_waits = 0u;
  std::uint64_t ready_set_member_storage = 0u;
  std::uint64_t ready_set_member_capacity = 0u;
  std::uint64_t many_validation_comparisons = 0u;
  std::uint64_t many_request_copies = 0u;
  std::uint64_t many_storage_growths = 0u;
  std::uint64_t ready_set_storage_growths = 0u;
};

[[nodiscard]] ReactorInvariantSnapshot
ValidateReactorCleanupInvariantsForTest() noexcept;

} // namespace rund::node

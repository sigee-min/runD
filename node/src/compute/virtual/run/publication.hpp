#pragma once

#include <rund/compute/status.hpp>

#include <array>
#include <cstddef>
#include <cstdint>
#include <limits>

namespace rund::compute::detail {

struct VirtualPipelineState;
struct VirtualRunTransaction;

// The publication cursor is the bounded, run-local capability for the two
// physical Pipeline banks.  It records the bases captured before a deferred
// Scan and the terminal count used to derive each private control generation;
// it never stores a logical-Q receipt array.
struct VirtualRunPublicationCursor final {
  static constexpr std::size_t BankCount = 2u;
  static constexpr std::uint64_t no_generation =
      std::numeric_limits<std::uint64_t>::max();

  std::array<std::uint64_t, BankCount> base_generation{};
  std::array<std::uint64_t, BankCount> base_payload_epoch{};
  std::array<std::uint8_t, BankCount> base_parity{};
  std::array<std::uint64_t, BankCount> control_generation{};
  std::array<std::uint64_t, BankCount> terminal_count{};
  bool active{};
};

[[nodiscard]] Status begin_virtual_run_publication_cursor(
    VirtualPipelineState &, VirtualRunTransaction &) noexcept;
[[nodiscard]] Status record_virtual_run_publication_terminal(
    VirtualRunTransaction &, std::size_t bank) noexcept;
[[nodiscard]] Status abort_virtual_run_publication_cursor(
    VirtualPipelineState &, VirtualRunPublicationCursor &) noexcept;
[[nodiscard]] Status poison_virtual_run_publication_cursor(
    VirtualPipelineState &, VirtualRunPublicationCursor &) noexcept;

} // namespace rund::compute::detail

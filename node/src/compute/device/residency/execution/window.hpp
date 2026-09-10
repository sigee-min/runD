#pragma once

#include "service.hpp"

#include <array>
#include <cstddef>
#include <cstdint>

namespace rund::compute::detail::residency::execution {

// Bounded Q<=4 coordinator. Internal Releases unlock recurrent service edges;
// one final WindowEvidence authenticates the already-consumed fixed journal.
class Window final {
public:
  [[nodiscard]] ExecutionLease begin(Authority &, const Plan &) noexcept;
  [[nodiscard]] bool issue(std::uint64_t epoch, Phase,
                           ExecutionTicket &) noexcept;
  [[nodiscard]] bool terminal(const ExecutionTicket &, ExecutionTerminal,
                              Status) noexcept;
  [[nodiscard]] bool release(const Release &) noexcept;
  [[nodiscard]] bool final(const WindowEvidence &) noexcept;
  // Integrity terminal after a backend-authenticated Known Final failed the
  // Compute join. It publishes no cache state and empties the complete
  // reserved journal. UnknownMayWrite is deliberately ineligible.
  [[nodiscard]] bool abandon_final(const WindowEvidence &) noexcept;
  [[nodiscard]] ExecutionClose close() noexcept;
  // Pre-native handoff rollback only. Bootstrap Input tickets may already be
  // issued or terminal, but no native Release or Output publication exists.
  // Every reserved row is conservatively invalidated.
  [[nodiscard]] bool abandon() noexcept;
  [[nodiscard]] WindowSnapshot snapshot() const noexcept;

  [[nodiscard]] const ExecutionLease &lease() const noexcept { return lease_; }

private:
  void advance_ready() noexcept;

  Authority *authority_{};
  const Plan *plan_{};
  ExecutionLease lease_{};
  Service service_{};
  std::array<Release, WindowCapacity> releases_{};
  std::array<bool, WindowCapacity> input_issued_{};
  std::array<bool, WindowCapacity> input_done_{};
  std::array<bool, WindowCapacity> output_issued_{};
  std::array<bool, WindowCapacity> output_done_{};
  std::size_t release_count_{};
  std::uint64_t release_prefix_{};
  std::uint64_t ready_prefix_{};
  std::uint64_t native_batches_{};
  std::uint64_t queue_calls_{};
  bool aborted_{};
  bool unknown_{};
  bool final_{};
  bool closed_{};
};

} // namespace rund::compute::detail::residency::execution

#pragma once

#include "../../registry.hpp"
#include "../../registry/sliding_owner.hpp"
#include "../sliding.hpp"

#include <algorithm>
#include <array>
#include <atomic>
#include <limits>
#include <mutex>
#include <new>

namespace rund::compute::detail::residency::execution {

enum class InputState : std::uint8_t {
  Free,
  Fetching,
  Completing,
  Ready,
  Promoting,
  Invalidating,
  Quarantined,
};
enum class OutputState : std::uint8_t {
  Free,
  Reserved,
  Draining,
  Completing,
  Ready,
  Persisting,
  Persisted,
  Invalidating,
  Quarantined,
};
enum class NativeState : std::uint8_t {
  Free,
  Promoting,
  Completing,
  Ready,
  Submitted,
  Draining,
  Invalidating,
  Quarantined,
};

extern std::atomic<std::uint64_t> NextSlidingOwner;

[[nodiscard]] bool add(std::uint64_t &value,
                       const std::uint64_t increment) noexcept;

[[nodiscard]] Status integrity_failure() noexcept;

struct Sliding::State final {
  struct InputCell final {
    SlidingCoordinate coordinate{};
    residency::PageKey key{};
    residency::PinInterval pin{};
    std::uint64_t turn{};
    std::uint64_t expected_bytes{};
    std::uint64_t next_use{NeverUse};
    Status completion{Status::success()};
    TerminalKind completion_terminal{TerminalKind::Known};
    std::uint64_t completion_bytes{};
    std::uint32_t physical_frame{std::numeric_limits<std::uint32_t>::max()};
    std::uint32_t use{};
    InputState state{InputState::Free};
    bool completion_may_write{};
    bool physical_handoff{};
  };
  struct OutputCell final {
    SlidingCoordinate coordinate{};
    std::uint64_t turn{};
    std::uint64_t expected_bytes{};
    std::uint64_t persist_sequence{NeverUse};
    std::uint32_t physical_frame{std::numeric_limits<std::uint32_t>::max()};
    std::uint32_t use{};
    OutputState state{OutputState::Free};
    Status completion{Status::success()};
    TerminalKind completion_terminal{TerminalKind::Known};
    std::uint64_t completion_bytes{};
    bool completion_may_write{};
    bool physical_handoff{};
    bool invalidate_after_terminal{};
  };
  struct NativeCell final {
    SlidingCoordinate coordinate{};
    std::array<std::uint32_t, WindowFootprintSourceCapacity> host_frames{};
    std::array<std::uint32_t, UseCapacity> device_input_frames{};
    std::array<std::uint32_t, UseCapacity> device_output_frames{};
    std::uint64_t turn{};
    std::uint64_t promote_bytes{};
    Status completion{Status::success()};
    TerminalKind completion_terminal{TerminalKind::Known};
    std::uint64_t completion_bytes{};
    std::uint32_t transfer_mask{};
    std::uint32_t source_count{};
    std::uint32_t input_count{};
    std::uint32_t output_count{};
    std::uint32_t resolved{};
    NativeState state{NativeState::Free};
    bool completion_may_write{};
    bool physical_handoff{};
    bool suppress{};
    bool invalidate_after_terminal{};
  };
  struct TerminalCell final {
    std::uint64_t ordinal{NeverUse};
    bool terminal{};
    bool may_write{};
    bool invalidating{};
  };

  explicit State(SlidingInvocation value, std::uint64_t authority_token,
                 std::uint64_t run_generation, std::uint64_t owner_nonce,
                 std::uint32_t input_capacity, std::uint32_t output_capacity,
                 std::size_t required_inputs, std::size_t required_outputs,
                 bool model) noexcept;

  [[nodiscard]] bool runnable() const noexcept;

  [[nodiscard]] bool credential(const SlidingTicket &ticket) const noexcept;

  [[nodiscard]] bool exact(const SlidingProjection &supplied,
                           const std::span<residency::PageUse> scratch,
                           SlidingProjection &expected) const noexcept;

  [[nodiscard]] bool accepts(const SlidingCoordinate coordinate) const noexcept;

  [[nodiscard]] InputCell *input(const SlidingTicket &ticket,
                                 const SlidingTicketKind kind) noexcept;

  [[nodiscard]] OutputCell *output(const SlidingTicket &ticket,
                                   const SlidingTicketKind kind) noexcept;

  [[nodiscard]] NativeCell *native(const SlidingTicket &ticket,
                                   const SlidingTicketKind kind) noexcept;

  [[nodiscard]] std::size_t
  host_bank(const SlidingCoordinate coordinate) const noexcept;

  [[nodiscard]] std::size_t
  input_offset(const SlidingCoordinate coordinate) const noexcept;

  [[nodiscard]] std::size_t
  output_offset(const SlidingCoordinate coordinate) const noexcept;

  [[nodiscard]] std::uint64_t
  bank_frontier(const SlidingCoordinate coordinate) const noexcept;

  [[nodiscard]] std::span<InputCell>
  input_ring(const SlidingCoordinate coordinate) noexcept;

  [[nodiscard]] std::span<OutputCell>
  output_ring(const SlidingCoordinate coordinate) noexcept;

  void advance_frontier() noexcept;

  [[nodiscard]] bool
  terminal_for(const SlidingCoordinate coordinate) const noexcept;

  void fail(SlidingCoordinate, Status, TerminalKind, bool) noexcept;
  void record_failure(SlidingCoordinate, Status, TerminalKind, bool) noexcept;
  void mark_unknown(Status) noexcept;
  void cancel_input_suffix(std::uint64_t) noexcept;
  void fail_native_suffix(std::uint64_t) noexcept;
  void fail_output_suffix(std::uint64_t) noexcept;
  void fail_terminal_suffix(std::uint64_t) noexcept;

  void advance_persist_frontier() noexcept;

  void retire(const SlidingCoordinate coordinate,
              const bool may_write) noexcept;

  void cancel_reserved(const SlidingCoordinate coordinate,
                       NativeCell &native) noexcept;

  void resolve_native(NativeCell &native) noexcept;

  [[nodiscard]] bool no_issued() const noexcept;

  [[nodiscard]] bool callbacks_quiesced() const noexcept;

  [[nodiscard]] bool rearm_bound(const residency::Identity next_plan,
                                 const std::uint64_t next_token,
                                 const std::uint64_t next_generation) noexcept;
  void clear_attempt() noexcept;
  [[nodiscard]] bool abandon_bound() noexcept;
  void quarantine_bound(Status) noexcept;

  void evidence(SlidingEvidence &result) const noexcept;

  mutable std::mutex gate;
  SlidingInvocation invocation{};
  residency::Identity plan{};
  std::array<InputCell, SlidingHostCellCapacity> inputs{};
  std::array<OutputCell, SlidingHostCellCapacity> outputs{};
  std::array<NativeCell, SlidingNativeCapacity> native_cells{};
  std::array<TerminalCell, SlidingNativeCapacity> terminals{};
  Status status{Status::success()};
  TerminalKind first_failure_terminal{TerminalKind::Known};
  TerminalKind terminal{TerminalKind::Known};
  SlidingCoordinate first_failure{};
  std::uint64_t token{};
  std::uint64_t generation{};
  std::uint64_t owner{};
  std::uint64_t planned{};
  std::uint64_t admitted{};
  std::uint64_t terminal_frontier{};
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
  std::uint32_t host_bank_count{};
  std::uint32_t input_stride{};
  std::uint32_t output_stride{};
  std::uint32_t input_count{};
  std::uint32_t output_count{};
  std::size_t coordinate_input_count{};
  std::size_t coordinate_output_count{};
  bool has_failure{};
  bool first_failure_may_write{};
  bool quarantine{};
  bool authority_bound{};
  bool model_only{};
  bool finalizing{};
  bool closed{};
  std::uint64_t final_nonce{};
};

} // namespace rund::compute::detail::residency::execution

#pragma once

#include <rund/compute/status.hpp>

#include <array>
#include <cstddef>
#include <cstdint>

namespace rund::compute::detail::residency::execution {

inline constexpr std::size_t FailureCapacity = 6u;
inline constexpr std::size_t WindowCapacity = 4u;

// Immutable final progress supplied by the one Authority execution journal.
// Receipt transports it; Receipt does not independently order epochs.
struct Progress final {
  std::uint64_t input_services{};
  std::uint64_t native_dispatches{};
  std::uint64_t native_completions{};
  std::uint64_t output_services{};
  std::uint64_t native_inflight_peak{};
};

enum class TerminalKind : std::uint8_t {
  Known,
  UnknownMayWrite,
};

// Bounded exact failure evidence from a recurrent native owner. `phases`
// names failed phases; `may_write` names issued target mutations; `terminal`
// names phases with a known terminal. Missing may-write terminals require
// quarantine regardless of the Status reason.
struct FailureEvidence final {
  std::uint64_t epoch{};
  std::uint8_t phases{};
  std::uint8_t may_write{};
  std::uint8_t terminal{};
};

// Adapter-owned final native evidence. It deliberately has no Host-service
// counters: only the run coordinator can authenticate Input supply and Output
// publication, then combine them into Evidence after both terminals.
struct NativeEvidence final {
  Status status{Status::fail(Reason::PipelineInvalid)};
  TerminalKind terminal{TerminalKind::Known};
  std::uint64_t plan_identity{};
  std::uint64_t token{};
  std::uint64_t generation{};
  std::uint64_t epoch_count{};
  std::uint64_t native_submissions{};
  std::uint64_t native_dispatches{};
  std::uint64_t native_completions{};
  std::uint64_t native_inflight_peak{};
  std::array<FailureEvidence, FailureCapacity> failures{};
  std::size_t failure_count{};
  std::uint64_t completed_ns{};
};

// One final-only evidence owner shared with a native recurrent consumer. It is
// fixed in Q and authenticated by Plan/token/generation. Compute receives it
// once; no native per-epoch callback crosses DeviceOps.
struct Evidence final {
  Status status{Status::fail(Reason::PipelineInvalid)};
  TerminalKind terminal{TerminalKind::Known};
  std::uint64_t plan_identity{};
  std::uint64_t token{};
  std::uint64_t generation{};
  std::uint64_t epoch_count{};
  std::uint64_t native_submissions{};
  Progress progress{};
  std::array<FailureEvidence, FailureCapacity> failures{};
  std::size_t failure_count{};
  std::uint64_t completed_ns{};
};

// One internal native bank-release receipt. This is neither a user callback
// nor final run evidence: Authority consumes it to unlock exact recurrent
// Host-service edges before the single public window terminal arrives.
struct Release final {
  Status status{Status::fail(Reason::PipelineInvalid)};
  TerminalKind terminal{TerminalKind::Known};
  std::uint64_t plan_identity{};
  std::uint64_t token{};
  std::uint64_t generation{};
  std::uint64_t epoch{};
  std::uint64_t backend_sequence{};
  std::uint8_t bank{};
  bool dispatched{};
  bool completed{};
  bool may_write{};
};

// One final native window terminal. `public_handoffs` is the runD/adapter
// boundary count. `native_batches` and `queue_calls` are actual backend facts
// and deliberately need not be equal on every backend.
struct WindowEvidence final {
  Status status{Status::fail(Reason::PipelineInvalid)};
  TerminalKind terminal{TerminalKind::Known};
  std::uint64_t plan_identity{};
  std::uint64_t token{};
  std::uint64_t generation{};
  std::uint64_t first_epoch{};
  std::uint64_t epoch_count{};
  std::uint64_t public_handoffs{};
  std::uint64_t native_batches{};
  std::uint64_t queue_calls{};
  std::uint64_t native_inflight_peak{};
  std::array<Release, WindowCapacity> releases{};
  std::size_t release_count{};
  std::uint64_t completed_ns{};
};

// One final for a native schedule that accepted every epoch before ready(0).
// Releases are authenticated incrementally through the fixed circular
// journal; this aggregate therefore carries no Q-sized receipt mirror.
struct ScheduleEvidence final {
  Status status{Status::fail(Reason::PipelineInvalid)};
  TerminalKind terminal{TerminalKind::Known};
  std::uint64_t plan_identity{};
  std::uint64_t token{};
  std::uint64_t generation{};
  std::uint64_t epoch_count{};
  std::uint64_t public_handoffs{};
  std::uint64_t native_batches{};
  std::uint64_t queue_calls{};
  std::uint64_t native_inflight_peak{};
  std::uint64_t released_prefix{};
  std::uint64_t completed_ns{};
};

struct WindowSnapshot final {
  std::uint64_t release_prefix{};
  std::uint64_t ready_prefix{};
  std::uint64_t native_batches{};
  std::uint64_t queue_calls{};
  bool aborted{};
  bool unknown{};
  bool final{};
};

} // namespace rund::compute::detail::residency::execution

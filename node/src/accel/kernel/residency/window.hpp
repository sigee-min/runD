#pragma once

#include "../callback.hpp"

#include <accel/check.hpp>

#include <array>
#include <cstddef>
#include <cstdint>
#include <memory>

namespace rund::node::accel::detail {

inline constexpr std::size_t ResidencyWindowCapacity = 4u;
inline constexpr std::size_t ResidencyWindowLocalCapacity = 32u;

// Policy-free backend transport for one bounded native window. Every batch
// owns its backend Pipeline strongly and copies its exact selected-local list;
// no caller span is retained past submit.
struct BackendResidencyWindowBatch final {
  std::shared_ptr<void> prepared{};
  std::array<std::uint32_t, ResidencyWindowLocalCapacity> locals{};
  std::size_t local_count{};
  std::uint64_t epoch{};
  std::uint32_t control_generation{};
  std::uint8_t bank{};
};

struct BackendResidencyWindowSignal final {
  // Success opens the exact native batch. Failure releases an already queued
  // batch through its backend no-write gate so a Host-service abort can drain
  // the fixed window without fabricating no-dispatch evidence or hanging a
  // queue wait. As with every AccelCheck in this layer, a failure reason is a
  // canonical process-lifetime reason literal, not caller-owned text.
  rund::AccelCheck admission{true, "ok"};
  std::uint64_t plan_identity{};
  std::uint64_t token{};
  std::uint64_t generation{};
  std::uint64_t epoch{};
  std::uint32_t control_generation{};
  std::uint8_t bank{};
};

// Exact accepted-window emergency drain. This is not cancellation: every
// command was already queued by the one public handoff. The backend opens all
// still-unsignalled no-write gates, conservatively resolves the failed window
// as UnknownMayWrite, and quarantines its native owner. `failure` is a
// canonical process-lifetime reason literal.
struct BackendResidencyWindowAbort final {
  rund::AccelCheck failure{false, "accel_kernel_pipeline_invalid"};
  std::uint64_t plan_identity{};
  std::uint64_t token{};
  std::uint64_t generation{};
};

struct BackendResidencyWindowReceipt final {
  rund::AccelCheck check{false, "accel_kernel_pipeline_invalid"};
  NativeTerminal terminal{NativeTerminal::Known};
  std::uint64_t epoch{};
  std::uint64_t backend_sequence{};
  std::uint8_t bank{};
  bool dispatched{};
  bool completed{};
  bool may_write{};
};

struct BackendResidencyWindowRelease final {
  BackendResidencyWindowReceipt receipt{};
  KernelResult result{};
};

struct BackendResidencyWindowFinal final {
  rund::AccelCheck check{false, "accel_kernel_pipeline_invalid"};
  NativeTerminal terminal{NativeTerminal::Known};
  std::uint64_t plan_identity{};
  std::uint64_t token{};
  std::uint64_t generation{};
  std::uint64_t first_epoch{};
  std::uint64_t public_handoffs{};
  std::uint64_t native_batches{};
  std::uint64_t queue_calls{};
  std::uint64_t native_inflight_peak{};
  std::array<BackendResidencyWindowReceipt, ResidencyWindowCapacity> receipts{};
  std::size_t receipt_count{};
  std::uint64_t completed_ns{};
};

using BackendResidencyWindowReleaseCompletion =
    void (*)(void *, BackendResidencyWindowRelease &&) noexcept;
using BackendResidencyWindowFinalCompletion =
    void (*)(void *, BackendResidencyWindowFinal &&) noexcept;

struct BackendResidencyWindowRequest final {
  std::uint64_t plan_identity{};
  std::uint64_t token{};
  std::uint64_t generation{};
  std::uint64_t first_epoch{};
  std::array<BackendResidencyWindowBatch, ResidencyWindowCapacity> batches{};
  std::size_t batch_count{};
  BackendResidencyWindowReleaseCompletion release{};
  BackendResidencyWindowFinalCompletion final{};
  void *user{};
};

} // namespace rund::node::accel::detail

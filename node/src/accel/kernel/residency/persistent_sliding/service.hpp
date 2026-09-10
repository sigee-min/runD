#pragma once

#include "control.hpp"
#include "request.hpp"

#include <accel/check.hpp>

#include <cstdint>

namespace rund::node::accel::detail {

struct PersistentResidencySlidingServiceIdentity final {
  std::uint64_t plan_identity{};
  std::uint64_t token{};
  std::uint64_t generation{};
  std::uint64_t coordinate{};
  std::uint64_t turn{};
  std::uint64_t descriptor_generation{};
  std::uint32_t control_generation{};
  std::uint8_t slot{};
};

// Host service publishes bytes and the exact descriptor before signalling an
// already-submitted native dependency. This operation never submits work.
struct PersistentResidencySlidingReadySignal final {
  PersistentResidencySlidingServiceIdentity identity{};
  rund::AccelCheck admission{true, "ok"};
};

// A synchronous service-thread wait request. It is not a backend callback.
struct PersistentResidencySlidingDoneWait final {
  PersistentResidencySlidingServiceIdentity identity{};
};

// Acquire-visible completion returned to the waiting service thread.
struct PersistentResidencySlidingDoneObservation final {
  PersistentResidencySlidingServiceIdentity identity{};
  rund::AccelCheck check{false, "accel_kernel_pipeline_invalid"};
  NativeTerminal terminal{NativeTerminal::Known};
  bool dispatched{};
  bool completed{};
  bool may_write{};
};

// Compute acknowledges a GPU completion only after its exact Drain/Persist
// service is terminal. Final is forbidden until every coordinate is
// acknowledged, so GPU done alone cannot publish output early.
struct PersistentResidencySlidingAcknowledgeDone final {
  PersistentResidencySlidingServiceIdentity identity{};
  rund::AccelCheck service_check{false, "accel_kernel_pipeline_invalid"};
};

// A Host data-service failure opens an already-submitted no-write suffix or
// quarantines it. It cannot enqueue replacement native work.
struct PersistentResidencySlidingServiceFailure final {
  PersistentResidencySlidingServiceIdentity identity{};
  rund::AccelCheck check{false, "accel_kernel_pipeline_invalid"};
  NativeTerminal terminal{NativeTerminal::Known};
  bool may_write{};
};

enum class PersistentResidencySlidingSubmitEvent : std::uint8_t {
  Declined,
  Accepted,
  Unknown,
  AcceptedUnknown,
};

// Backend submit returns ownership of acceptance to the common service.  It
// may report a terminal-pending failure after an accepted native prefix, but
// it never invokes Final or clears Control itself.
struct PersistentResidencySlidingSubmitResult final {
  rund::AccelCheck status{false, "accel_kernel_pipeline_invalid"};
  PersistentResidencySlidingSubmitEvent event{
      PersistentResidencySlidingSubmitEvent::Declined};
};

using PersistentResidencySlidingSubmitResultFn =
    PersistentResidencySlidingSubmitResult (*)(
        const PersistentResidencySlidingRequest &,
        PersistentResidencySlidingControl &) noexcept;
using PersistentResidencySlidingWaitDone =
    rund::AccelCheck (*)(PersistentResidencySlidingControl &,
                         const PersistentResidencySlidingDoneWait &,
                         PersistentResidencySlidingDoneObservation &) noexcept;
using PersistentResidencySlidingSignalReady = rund::AccelCheck (*)(
    PersistentResidencySlidingControl &,
    const PersistentResidencySlidingReadySignal &) noexcept;
using PersistentResidencySlidingAckDone = rund::AccelCheck (*)(
    PersistentResidencySlidingControl &,
    const PersistentResidencySlidingAcknowledgeDone &) noexcept;
using PersistentResidencySlidingFailService = rund::AccelCheck (*)(
    PersistentResidencySlidingControl &,
    const PersistentResidencySlidingServiceFailure &) noexcept;
using PersistentResidencySlidingQuarantineUnknown = void (*)(
    PersistentResidencySlidingControl &) noexcept;

struct PersistentResidencySlidingServiceOps final {
  PersistentResidencySlidingSubmitResultFn submit_result{};
  PersistentResidencySlidingWaitDone wait_done{};
  PersistentResidencySlidingSignalReady signal_ready{};
  PersistentResidencySlidingAckDone ack_done{};
  PersistentResidencySlidingFailService fail_service{};
  PersistentResidencySlidingQuarantineUnknown quarantine_unknown{};
};

} // namespace rund::node::accel::detail

#pragma once

#include "projection.hpp"

#include <rund/compute/virtual.hpp>

#include <mutex>

namespace rund::compute::detail {

struct VirtualRunAdmission final {
  bool transaction_window{};
  bool transaction_scan{};
  bool try_recurrent{};
  Status status{Status::fail(Reason::PipelineInvalid)};
  bool claim{};
  bool poison_pipeline{};
};

enum class VirtualRunAdmissionMode : std::uint8_t {
  Synchronous,
  AsynchronousDeviceVsm,
};

// Admission is a pure projection of already sealed run facts. It does not
// prepare a backend owner, begin a transaction, acquire a lease, or mutate
// backing, pool, or Authority state.
[[nodiscard]] VirtualRunAdmission
admit_virtual_run(const VirtualPipelineState &, const VirtualRunProjection &,
                  const VirtualBacking &, const VirtualBacking &) noexcept;

// Common lifecycle admission for sync and async execution. The lock is
// acquired here, but the Running phase is claimed by the caller only after
// any caller-owned setup allocation has succeeded. A failed capability probe
// therefore still returns `claim=true` so the caller can publish its normal
// run evidence and terminal.
[[nodiscard]] VirtualRunAdmission
admit_virtual_run(const std::shared_ptr<VirtualPipelineState> &, std::uint64_t,
                  std::unique_lock<std::mutex> &, VirtualRunAdmissionMode) noexcept;

} // namespace rund::compute::detail

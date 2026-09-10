#pragma once

#include "device_vsm/route/proof.hpp"
#include "projection.hpp"
#include "transaction.hpp"

#include "../../device/residency/execution/owner.hpp"
#include "../../device/residency/execution/plan.hpp"

#include <rund/compute/stats.hpp>

#include <array>
#include <cstdint>
#include <limits>
#include <memory>
#include <optional>
#include <utility>

namespace rund::compute::detail {

// Seals the Direct pointwise invocation into the one backend-neutral
// recurrent execution authority. This function projects only immutable run
// facts; it does not admit frames or choose a backend route.
[[nodiscard]] residency::execution::SealResult
seal_virtual_execution(const VirtualPipelineState &,
                       const VirtualRunProjection &) noexcept;

struct VirtualExecutionPrepared final {
  residency::execution::Plan plan{};
  residency::execution::Owner owner{};

  [[nodiscard]] explicit operator bool() const noexcept {
    return plan.identity() != 0u && plan.epoch_count() == 1u && owner;
  }
};

// Cold-owner lookup is performed before Authority admission. Unsupported
// adapters therefore fall back to the existing epoch runner without rolling
// back an execution journal or touching backing state.
[[nodiscard]] Status
prepare_virtual_execution(VirtualPipelineState &, const VirtualRunProjection &,
                          VirtualExecutionPrepared &) noexcept;

enum class VirtualExecutionDisposition : std::uint8_t {
  // Every execution result is terminal unless the producer proves a clean
  // pre-native capability decline at its exact preparation boundary.
  Terminal,
  CleanPreNativeDecline,
};

struct VirtualExecutionResult final {
  Status status{Status::fail(Reason::PipelineInvalid)};
  std::uint64_t failed_page{ResidencyStats::no_failed_page};
  std::uint64_t output_hash{};
  bool poison_pipeline{};
  std::optional<VirtualRunWriteCertainty> certainty{};
  VirtualExecutionDisposition disposition{
      VirtualExecutionDisposition::Terminal};
};

struct VirtualExecutionWindowPrepared final {
  residency::execution::Plan plan{};
  std::array<std::shared_ptr<PipelineState>, residency::execution::BankCapacity>
      pipelines{};

  [[nodiscard]] explicit operator bool() const noexcept {
    return plan.identity() != 0u && plan.epoch_count() >= 2u &&
           pipelines[0u] != nullptr && pipelines[1u] != nullptr;
  }
};

// Cold Q-independent owner for the Direct physical Sliding path.  The
// source-private state retains the exact prepared parity roles and native
// fixed-W controller; no Q-sized batch or callback storage crosses this API.
struct VirtualExecutionSlidingPrepared final {
  using Cleanup = void (*)(std::shared_ptr<void> &, std::uint64_t) noexcept;

  residency::execution::Plan plan{};
  std::shared_ptr<void> owner{};
  Cleanup cleanup{};
  std::uint64_t lease_token{};

  VirtualExecutionSlidingPrepared() = default;
  VirtualExecutionSlidingPrepared(const VirtualExecutionSlidingPrepared &) =
      delete;
  VirtualExecutionSlidingPrepared &
  operator=(const VirtualExecutionSlidingPrepared &) = delete;

  VirtualExecutionSlidingPrepared(
      VirtualExecutionSlidingPrepared &&other) noexcept
      : plan(std::move(other.plan)), owner(std::move(other.owner)),
        cleanup(other.cleanup), lease_token(other.lease_token) {
    other.cleanup = nullptr;
    other.lease_token = 0u;
    other.plan = {};
  }

  VirtualExecutionSlidingPrepared &
  operator=(VirtualExecutionSlidingPrepared &&other) noexcept {
    if (this != &other) {
      reset();
      plan = std::move(other.plan);
      owner = std::move(other.owner);
      cleanup = other.cleanup;
      lease_token = other.lease_token;
      other.cleanup = nullptr;
      other.lease_token = 0u;
      other.plan = {};
    }
    return *this;
  }

  ~VirtualExecutionSlidingPrepared() { reset(); }

  void reset() noexcept {
    if (cleanup != nullptr) {
      cleanup(owner, lease_token);
    }
    cleanup = nullptr;
    owner.reset();
    plan = {};
    lease_token = 0u;
  }

  [[nodiscard]] explicit operator bool() const noexcept {
    return plan.identity() != 0u && plan.epoch_count() >= 2u &&
           owner != nullptr;
  }
};

enum class VirtualDeviceVsmPreparedOrigin : std::uint8_t {
  ColdNew,
  WarmCached,
};

// Cold aggregate owner for the pointwise or centered-Window page-coordinate
// DeviceVsm product. Unlike the Sliding owner, this route exposes no
// per-coordinate Host service surface: the prepared GPU owner receives the
// complete page recurrence in one submit.
struct VirtualExecutionDeviceVsmPrepared final {
  std::shared_ptr<void> owner{};
  const char *reason{"compute_backend_unsupported"};
  // Immutable route proof copied at the side-effecting preparation boundary.
  // The lower owner remains responsible for credential and mapping checks.
  VirtualDeviceVsmRouteProof proof{};
  VirtualDeviceVsmPreparedOrigin origin{
      VirtualDeviceVsmPreparedOrigin::ColdNew};
  // Set only after the backend reports that warm-owner state was mutated.
  // An unsuccessful validation/rearm with no mutation keeps the submitted
  // warm cache reusable; mutated failures take the cold cleanup path.
  bool rearm_mutated{};

  [[nodiscard]] explicit operator bool() const noexcept {
    return owner != nullptr && proof.page_count() >= 2u;
  }
};

[[nodiscard]] VirtualExecutionResult execute_virtual_execution_device_vsm(
    VirtualPipelineState &, std::span<VirtualBacking *const>, VirtualBacking &,
    const VirtualRunProjection &, const VirtualExecutionDeviceVsmPrepared &,
    Stats &) noexcept;

struct VirtualAsyncStart final {
  std::shared_ptr<void> frame{};
  Status status{Status::fail(Reason::BackendUnsupported)};
  bool submitted{};
};

using VirtualWake = void (*)(void *) noexcept;

[[nodiscard]] VirtualAsyncStart
start_virtual_pipeline_async(const std::shared_ptr<VirtualPipelineState> &,
                             void *, VirtualWake) noexcept;

[[nodiscard]] Status resume_virtual_pipeline_async(std::shared_ptr<void> &,
                                                   Stats &) noexcept;

// Mutation-free Direct/HostCoherent preparation. Unsupported topology,
// incomplete tail materialization, noncoherent memory, or a backend without
// an authenticated descriptor gate returns BackendUnsupported before an
// Authority generation or backing recovery marker exists.
[[nodiscard]] Status
prepare_virtual_execution_sliding(VirtualPipelineState &,
                                  const VirtualRunProjection &,
                                  VirtualExecutionSlidingPrepared &) noexcept;

// One physical Authority generation and one raw fixed-W native owner for the
// whole Direct run. Host service is callback-return authenticated; the caller
// performs no epoch loop and observes one Final.
[[nodiscard]] VirtualExecutionResult execute_virtual_execution_sliding(
    VirtualPipelineState &, VirtualBacking &, VirtualBacking &,
    const VirtualRunProjection &, const VirtualExecutionSlidingPrepared &,
    Stats &) noexcept;

// Mutation-free native-window admission. It runs before Authority/backing
// mutation and returns BackendUnsupported for adapters without an actual
// bounded selected-command owner, preserving the rolling fallback.
[[nodiscard]] Status
prepare_virtual_execution_window(VirtualPipelineState &,
                                 const VirtualRunProjection &,
                                 VirtualExecutionWindowPrepared &) noexcept;

// Q>=2 Direct pointwise execution. Q<=4 is one bounded native window; larger
// runs use the fixed recurrent Stream controller and still expose only one
// public Authority/backing terminal.
[[nodiscard]] VirtualExecutionResult
execute_virtual_execution_window(VirtualPipelineState &, VirtualBacking &,
                                 VirtualBacking &, const VirtualRunProjection &,
                                 const VirtualExecutionWindowPrepared &,
                                 Stats &) noexcept;

// Consumes the Q=1 compound execution transaction end to end. The caller
// holds both backing gates and the Pool execution gate for the full join.
[[nodiscard]] VirtualExecutionResult
execute_virtual_execution(VirtualPipelineState &, VirtualBacking &,
                          VirtualBacking &, const VirtualRunProjection &,
                          const VirtualExecutionPrepared &, Stats &) noexcept;

// Focused contract hook. The next physical Virtual coordinator stops after
// its successful Output backing stage but before final Authority close. This
// proves emergency journal cleanup cannot publish or clear backing recovery.
void inject_virtual_execution_close_failure_once() noexcept;
[[nodiscard]] bool consume_virtual_execution_close_failure_once() noexcept;

} // namespace rund::compute::detail

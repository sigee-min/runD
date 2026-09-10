#pragma once

#include "window.hpp"

#include <array>
#include <cstddef>
#include <cstdint>
#include <memory>

namespace rund::node::accel::detail {

// Four logical roles are the closed product of two recurrent banks and two
// publication parities. A backend may observe aliased prepared owners, but it
// must not collapse the logical role identity or derive policy from pointers.
inline constexpr std::size_t ResidencyScheduleRoleCapacity = 4u;

enum class BackendResidencyScheduleLowering : std::uint8_t {
  Unsupported,
  VulkanTimeline,
  MetalPending,
};

// Cold preparation result. `owner` is either an already retained fixed
// lowering (Metal) or empty when the backend needs only transient submit-time
// descriptors (Vulkan).
struct BackendResidencySchedulePreparation final {
  rund::AccelCheck check{false, "accel_kernel_pipeline_invalid"};
  BackendResidencyScheduleLowering kind{
      BackendResidencyScheduleLowering::Unsupported};
  std::uint64_t retained_bytes{};
  std::uint64_t transient_bytes{};
  std::uint64_t queue_calls{};
  // A cold backend lowering is part of the capability result. Compute holds it
  // behind an admitted storage reservation before Authority can be acquired.
  std::shared_ptr<void> owner{};
  bool callbacks_async{};
};

struct BackendResidencyScheduleRole final {
  std::shared_ptr<void> prepared{};
  std::array<std::uint32_t, ResidencyWindowLocalCapacity> locals{};
  std::size_t local_count{};
  std::uint32_t first_control_generation{};
  std::uint32_t control_generation_stride{};
  std::uint8_t role{};
  std::uint8_t bank{};
};

struct BackendResidencyScheduleRelease final {
  BackendResidencyWindowReceipt receipt{};
  KernelResult result{};
};

// A whole-schedule Final is algebraic. It authenticates a completed prefix and
// one optional accepted no-write interval instead of retaining Q receipts.
struct BackendResidencyScheduleFinal final {
  rund::AccelCheck check{false, "accel_kernel_pipeline_invalid"};
  NativeTerminal terminal{NativeTerminal::Known};
  std::uint64_t plan_identity{};
  std::uint64_t token{};
  std::uint64_t generation{};
  std::uint64_t epoch_count{};
  std::uint64_t public_handoffs{};
  std::uint64_t native_batches{};
  std::uint64_t queue_calls{};
  std::uint64_t native_inflight_peak{};
  std::uint64_t released_prefix{};
  std::uint64_t completed_prefix{};
  std::uint64_t suppressed_first{};
  std::uint64_t suppressed_count{};
  std::uint64_t completed_ns{};
};

using BackendResidencyScheduleReleaseCompletion =
    void (*)(void *, BackendResidencyScheduleRelease &&) noexcept;
using BackendResidencyScheduleFinalCompletion =
    void (*)(void *, BackendResidencyScheduleFinal &&) noexcept;

// Policy-free native recurrence. The role for epoch e is e mod role_count;
// the last epoch consumes tail_local_count and every other epoch consumes the
// role's local_count. No Q-entry policy or callback array crosses this seam.
struct BackendResidencyScheduleRequest final {
  std::uint64_t plan_identity{};
  std::uint64_t token{};
  std::uint64_t generation{};
  std::uint64_t epoch_count{};
  std::array<BackendResidencyScheduleRole, ResidencyScheduleRoleCapacity>
      roles{};
  std::size_t role_count{};
  std::size_t tail_local_count{};
  std::shared_ptr<void> lowering{};
  // Compute-owned exact storage reservation. Backends do not inspect it; the
  // final delivery retains it until the callback has returned.
  std::shared_ptr<void> admission{};
  BackendResidencyScheduleReleaseCompletion release{};
  BackendResidencyScheduleFinalCompletion final{};
  void *user{};
};

} // namespace rund::node::accel::detail

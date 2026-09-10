#pragma once

#include "../local.hpp"

#include <array>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <mutex>
#include <span>
#include <vector>

namespace rund::node::accel::detail::schedule_detail {

#if defined(__APPLE__) && defined(RUND_NODE_HAVE_METAL_SDK)

#if defined(__MAC_26_0) && defined(__MAC_OS_X_VERSION_MAX_ALLOWED) &&          \
    __MAC_OS_X_VERSION_MAX_ALLOWED >= __MAC_26_0

inline constexpr std::uint64_t ScheduleMagic = 0x4d544c5343484544ull;
inline constexpr std::uint64_t TerminalDeadlineNs = 30u * NSEC_PER_SEC;
inline constexpr std::uint64_t FaultTerminalDeadlineNs = 10u * NSEC_PER_MSEC;

struct MetalResidencyScheduleOwner;

struct MetalResidencyScheduleDelivery final {
  std::shared_ptr<MetalResidencyScheduleOwner> owner{};
  BackendResidencyScheduleRequest callbacks{};
  BackendResidencyScheduleRelease release{};
  BackendResidencyScheduleFinal final{};
  bool emit_final{};
};

struct MetalResidencyScheduleCommand final {
  id<MTL4CommandAllocator> allocator = nil;
  id<MTL4CommandBuffer> command = nil;
  MetalSequence *sequence{};
  std::uint64_t epoch{};
  std::uint64_t ready_value{};
  std::uint64_t done_value{};
  std::uint64_t dispatch_count{};
  std::uint64_t control_count{};
  std::uint64_t reset_count{};
  std::uint64_t reset_bytes{};
  std::uint32_t control_generation{};
  rund::AccelCheck admission{true, "ok"};
  KernelResult result{};
  MetalResidencyScheduleDelivery delivery{};
  bool signaled{};
  bool completed{};
  bool force_device_lost{};
  bool force_terminal_loss{};
};

struct MetalResidencyScheduleOwner final
    : std::enable_shared_from_this<MetalResidencyScheduleOwner> {
  ~MetalResidencyScheduleOwner();

  std::mutex gate{};
  std::uint64_t magic{ScheduleMagic};
  MetalAdapter *adapter{};
  id<MTL4CommandQueue> queue = nil;
  dispatch_queue_t service = nil;
  std::vector<MetalResidencyScheduleCommand> commands{};
  std::array<BackendResidencyScheduleRole, ResidencyScheduleRoleCapacity>
      roles{};
  std::size_t tail_local_count{};
  BackendResidencyScheduleRequest request{};
  std::shared_ptr<void> quarantine{};
  rund::AccelCheck first_failure{true, "ok"};
  rund::AccelCheck abort_failure{false, "compute_backend_failed"};
  std::uint64_t release_count{};
  std::uint64_t success_prefix{};
  std::uint64_t suppressed_first{};
  std::uint64_t suppressed_count{};
  std::uint64_t native_inflight{};
  std::uint64_t native_inflight_peak{};
  std::uint64_t submit_begin_ns{};
  std::uint64_t retained_bytes{};
  bool active{};
  bool aborting{};
  bool unknown{};
  bool final_sent{};
};

[[nodiscard]] bool add(std::uint64_t left, std::uint64_t right,
                       std::uint64_t &result) noexcept;
[[nodiscard]] bool same_role(const BackendResidencyScheduleRole &left,
                             const BackendResidencyScheduleRole &right);
[[nodiscard]] bool control_generation(
    const BackendResidencyScheduleRole &role, std::uint64_t epoch,
    std::uint32_t &generation) noexcept;
[[nodiscard]] bool valid_roles(
    std::span<const BackendResidencyScheduleRole> roles,
    std::uint64_t epoch_count, MetalAdapter *&adapter) noexcept;
[[nodiscard]] std::shared_ptr<MetalResidencyScheduleOwner>
owner_of(const std::shared_ptr<void> &opaque) noexcept;

void deliver(void *raw) noexcept;
void complete(const std::shared_ptr<MetalResidencyScheduleOwner> &owner,
              std::uint64_t epoch, NativeTerminal terminal) noexcept;
void arm_timeout(const std::shared_ptr<MetalResidencyScheduleOwner> &owner,
                 std::uint64_t epoch, std::uint64_t delay) noexcept;
[[nodiscard]] bool signal_locked(
    const std::shared_ptr<MetalResidencyScheduleOwner> &owner,
    MetalResidencyScheduleCommand &entry,
    const rund::AccelCheck admission) noexcept;

#endif

#endif

} // namespace rund::node::accel::detail::schedule_detail

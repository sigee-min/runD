#pragma once

#include "../../../../kernel/residency/window.hpp"
#include "../../../../kernel/terminal.hpp"

#include <accel/check.hpp>

#include <array>
#include <atomic>
#include <cstdint>
#include <memory>
#include <mutex>

#if defined(__APPLE__) && defined(RUND_NODE_HAVE_METAL_SDK)
#import <Metal/Metal.h>
#include <dispatch/dispatch.h>
#endif

namespace rund::node::accel::detail {

#if defined(__APPLE__) && defined(RUND_NODE_HAVE_METAL_SDK)

enum class MetalResidencySubmissionPhase : std::uint8_t {
  Unsupported,
  Cold,
  Ready,
  Failed,
  Quarantined,
};

struct MetalResidencySubmission final {
#if defined(__MAC_26_0) && defined(__MAC_OS_X_VERSION_MAX_ALLOWED) &&          \
    __MAC_OS_X_VERSION_MAX_ALLOWED >= __MAC_26_0
  id<MTL4CommandAllocator> allocator = nil;
  id<MTL4CommandBuffer> command = nil;
  id<MTLResidencySet> resources = nil;
  id<MTLSharedEvent> event = nil;
  MTLSharedEventListener *listener = nil;
  dispatch_source_t watchdog = nil;
#endif
  std::weak_ptr<void> owner{};
  TerminalCell terminal{};
  // A queued dispatch-source event can outlive its original generation. Zero
  // means that no deadline is published; otherwise a handler must prove the
  // current monotonic time reached this armed generation's deadline.
  std::atomic<std::uint64_t> watchdog_deadline_ns{};
  MetalResidencySubmissionPhase phase =
      MetalResidencySubmissionPhase::Unsupported;
  std::uint64_t event_value{};
  std::uint64_t submit_begin_ns{};
  std::uint64_t dispatch_count{};
  std::uint64_t control_count{};
  std::uint64_t reset_count{};
  std::uint64_t reset_bytes{};
  std::uint64_t native_submit_count{1u};
  std::uint64_t command_inflight_peak{};
  bool force_device_lost{};
  bool force_terminal_loss{};

  [[nodiscard]] bool supported() const noexcept {
    return phase != MetalResidencySubmissionPhase::Unsupported;
  }

  [[nodiscard]] bool ready() const noexcept {
    return phase == MetalResidencySubmissionPhase::Ready;
  }
};

// Word-addressed to keep the Host/MSL ABI independent of nested-structure
// padding.  The first row is the Host expectation, the second row is the
// release-published row consumed by the GPU, and the tail is GPU-only result
// evidence observed after the exact shared-event terminal.
inline constexpr std::size_t MetalResidencySlidingRowWords = 58u;
inline constexpr std::size_t MetalResidencySlidingLocalWord = 26u;
inline constexpr std::size_t MetalResidencySlidingPayloadWords = 120u;
inline constexpr std::size_t MetalResidencySlidingPublishedWord =
    MetalResidencySlidingRowWords;
inline constexpr std::size_t MetalResidencySlidingAcceptedWord =
    MetalResidencySlidingRowWords * 2u;
inline constexpr std::size_t MetalResidencySlidingReasonWord =
    MetalResidencySlidingAcceptedWord + 1u;
inline constexpr std::size_t MetalResidencySlidingObservedGenerationWord =
    MetalResidencySlidingReasonWord + 1u;

struct MetalResidencySlidingPayload final {
  std::array<std::uint32_t, MetalResidencySlidingPayloadWords> words{};
};

static_assert(sizeof(MetalResidencySlidingPayload) == 480u);

struct MetalResidencySlidingSubmissionStorage final {
  MetalResidencySlidingPayload payload{};
};

// One cold fixed gate per prepared Pipeline.  It authenticates an immutable
// already-selected row and controls only the existing device guard; no page,
// replacement, next-use, or Graph policy crosses this boundary.
struct MetalResidencySlidingGate final {
#if defined(__MAC_26_0) && defined(__MAC_OS_X_VERSION_MAX_ALLOWED) &&          \
    __MAC_OS_X_VERSION_MAX_ALLOWED >= __MAC_26_0
  id<MTLBuffer> descriptor = nil;
  id<MTLComputePipelineState> pipeline = nil;
  id<MTLIndirectCommandBuffer> command = nil;
  id<MTLSharedEvent> ready = nil;
#endif
  std::array<std::uint32_t, MetalResidencySlidingRowWords> published_row{};
  std::shared_ptr<void> active_owner{};
  std::shared_ptr<void> quarantine_owner{};
  std::atomic_bool force_stale_once{false};
  const void *owner{};
  std::uint64_t plan_identity{};
  std::uint64_t token{};
  std::uint64_t run_generation{};
  std::uint64_t last_coordinate{};
  std::uint64_t last_turn{};
  std::uint64_t last_descriptor_generation{};
  std::uint64_t next_ready_value{1u};
  std::uint64_t retained_bytes{};
  std::atomic<std::uint64_t> gpu_result_read_count{};
  std::uint8_t stride{};
  std::uint8_t slot{};
  bool authenticated{};
  bool active{};
  bool ready_for_submit{};
  std::atomic_bool quarantined{false};
};

enum class MetalResidencyWindowPhase : std::uint8_t {
  Unsupported,
  Cold,
  Ready,
  Quarantined,
};

struct MetalResidencyWindowCommand final {
#if defined(__MAC_26_0) && defined(__MAC_OS_X_VERSION_MAX_ALLOWED) &&          \
    __MAC_OS_X_VERSION_MAX_ALLOWED >= __MAC_26_0
  id<MTL4CommandAllocator> allocator = nil;
  id<MTL4CommandBuffer> command = nil;
  dispatch_source_t watchdog = nil;
#endif
  TerminalCell terminal{};
  std::atomic<std::uint64_t> watchdog_deadline_ns{};
  std::weak_ptr<void> leader{};
  std::uint64_t terminal_generation{};
  std::uint64_t ready_value{};
  std::uint64_t done_value{};
  std::uint64_t dispatch_count{};
  std::uint64_t control_count{};
  std::uint64_t reset_count{};
  std::uint64_t reset_bytes{};
  std::size_t batch_index{};
  rund::AccelCheck admission{true, "ok"};
  bool force_device_lost{};
  bool force_terminal_loss{};
  bool force_abort_unknown{};
};

struct MetalResidencyWindowDelivery final {
  BackendResidencyWindowRequest callbacks{};
  std::array<BackendResidencyWindowRelease, ResidencyWindowCapacity> releases{};
  std::size_t release_count{};
  BackendResidencyWindowFinal final{};
  bool emit_final{};
};

struct MetalResidencyWindowRun final {
  std::mutex gate{};
  BackendResidencyWindowRequest request{};
  std::array<BackendResidencyWindowReceipt, ResidencyWindowCapacity> receipts{};
  std::array<KernelResult, ResidencyWindowCapacity> results{};
  std::array<bool, ResidencyWindowCapacity> completed{};
  std::size_t release_count{};
  std::uint64_t native_inflight{};
  std::uint64_t native_inflight_peak{};
  std::uint64_t submit_begin_ns{};
  bool active{};
  bool unknown{};
  bool final_sent{};
};

struct MetalResidencyWindow final {
#if defined(__MAC_26_0) && defined(__MAC_OS_X_VERSION_MAX_ALLOWED) &&          \
    __MAC_OS_X_VERSION_MAX_ALLOWED >= __MAC_26_0
  id<MTLSharedEvent> ready = nil;
  id<MTLSharedEvent> done = nil;
  MTLSharedEventListener *listener = nil;
  dispatch_queue_t service = nil;
#endif
  std::array<MetalResidencyWindowCommand, 2u> commands{};
  std::array<MetalResidencyWindowDelivery, 2u> deliveries{};
  std::weak_ptr<void> owner{};
  MetalResidencyWindowRun run{};
  MetalResidencyWindowPhase phase{MetalResidencyWindowPhase::Unsupported};
  std::uint64_t event_value{};

  [[nodiscard]] bool ready_for_submit() const noexcept {
    return phase == MetalResidencyWindowPhase::Ready;
  }
};

#endif

} // namespace rund::node::accel::detail

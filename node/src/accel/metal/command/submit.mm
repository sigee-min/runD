#include <accel/check.hpp>

#include "../../clock.hpp"
#include "../adapter.hpp"
#include "submit.hpp"
#include <cstdint>

#if defined(__APPLE__) && defined(RUND_NODE_HAVE_METAL_SDK)
#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#endif

namespace rund::node::accel::detail {

#if defined(__APPLE__) && defined(RUND_NODE_HAVE_METAL_SDK)
namespace {

[[nodiscard]] const char *
MetalCommandFailureReason(id<MTLCommandBuffer> const command) noexcept {
  NSError *const error = command == nil ? nil : [command error];
#if defined(__MAC_OS_X_VERSION_MAX_ALLOWED) && defined(__MAC_10_13) &&         \
    __MAC_OS_X_VERSION_MAX_ALLOWED >= __MAC_10_13
  if (error != nil &&
      [error.domain isEqualToString:MTLCommandBufferErrorDomain]) {
    switch (error.code) {
    case MTLCommandBufferErrorDeviceRemoved:
      return "compute_device_lost";
    case MTLCommandBufferErrorOutOfMemory:
      return "compute_device_capacity";
    default:
      break;
    }
  }
#else
  static_cast<void>(error);
#endif
  return "accel_metal_command_unavailable";
}

} // namespace
#endif

rund::AccelCheck WaitCommand(MetalAdapter &adapter, void *const command_buffer,
                             rund::RuntimeStats *const stats) {
#if defined(__APPLE__) && defined(RUND_NODE_HAVE_METAL_SDK)
  if (stats != nullptr) {
    *stats = rund::RuntimeStats{.ok = true, .reason = "ok"};
  }
  id<MTLCommandBuffer> command = (__bridge id<MTLCommandBuffer>)command_buffer;
  if (command == nil) {
    SetMetalLastError(adapter, "accel_metal_command_unavailable");
    return rund::AccelCheck{false, "accel_metal_command_unavailable"};
  }
  const std::uint64_t submit_begin = MonotonicNanoseconds();
  const bool force_device_lost =
      adapter.fault_device_lost_once.exchange(false, std::memory_order_relaxed);
  [command commit];
  [command waitUntilCompleted];
  const std::uint64_t submit_wait_ns = MonotonicNanoseconds() - submit_begin;
  RecordMetalCommandSubmitWaitNs(adapter, submit_wait_ns);
  if (stats != nullptr) {
    stats->command_submit_count = 1u;
    stats->command_submit_wait_ns = submit_wait_ns;
  }
  if (force_device_lost) {
    SetMetalLastError(adapter, "compute_device_lost");
    return rund::AccelCheck{false, "compute_device_lost"};
  }
  if ([command status] != MTLCommandBufferStatusCompleted ||
      [command error] != nil) {
    const char *const reason = MetalCommandFailureReason(command);
    SetMetalLastError(adapter, reason);
    return rund::AccelCheck{false, reason};
  }
  const std::uint64_t kernel_ns = RecordMetalComputeKernelSeconds(
      adapter, [command GPUStartTime], [command GPUEndTime]);
  if (stats != nullptr && kernel_ns != 0u) {
    stats->accel_kernel_ns = kernel_ns;
    stats->accel_timestamp_count = 1u;
    stats->accel_timestamp_source = "metal_command_buffer_compute_time";
  }
  return rund::AccelCheck{true, "ok"};
#else
  (void)adapter;
  (void)command_buffer;
  (void)stats;
  return rund::AccelCheck{false, "accel_metal_unavailable"};
#endif
}

rund::AccelCheck QueueCommand(MetalAdapter &adapter, void *const command_buffer,
                              const KernelCompletion completion,
                              void *const user) noexcept {
#if defined(__APPLE__) && defined(RUND_NODE_HAVE_METAL_SDK)
  id<MTLCommandBuffer> command = (__bridge id<MTLCommandBuffer>)command_buffer;
  if (command == nil || completion == nullptr) {
    SetMetalLastError(adapter, "accel_metal_command_unavailable");
    return rund::AccelCheck{false, "accel_metal_command_unavailable"};
  }
  MetalAdapter *const target = &adapter;
  const bool force_device_lost =
      adapter.fault_device_lost_once.exchange(false, std::memory_order_relaxed);
  const std::uint64_t submit_begin = MonotonicNanoseconds();
  [command addCompletedHandler:^(id<MTLCommandBuffer> finished) {
    const std::uint64_t submit_wait_ns = MonotonicNanoseconds() - submit_begin;
    RecordMetalCommandSubmitWaitNs(*target, submit_wait_ns);
    rund::RuntimeStats stats{
        .command_submit_count = 1u,
        .command_submit_wait_ns = submit_wait_ns,
        .ok = true,
        .reason = "ok",
    };
    if (force_device_lost) {
      SetMetalLastError(*target, "compute_device_lost");
      completion(user,
                 KernelResult{
                     .check = rund::AccelCheck{false, "compute_device_lost"},
                     .stats = stats,
                 });
      return;
    }
    if ([finished status] != MTLCommandBufferStatusCompleted ||
        [finished error] != nil) {
      const char *const reason = MetalCommandFailureReason(finished);
      SetMetalLastError(*target, reason);
      completion(user, KernelResult{
                           .check = rund::AccelCheck{false, reason},
                           .stats = stats,
                       });
      return;
    }
    stats.accel_kernel_ns = RecordMetalComputeKernelSeconds(
        *target, [finished GPUStartTime], [finished GPUEndTime]);
    if (stats.accel_kernel_ns != 0u) {
      stats.accel_timestamp_count = 1u;
      stats.accel_timestamp_source = "metal_command_buffer_compute_time";
    }
    completion(user, KernelResult{
                         .check = rund::AccelCheck{true, "ok"},
                         .stats = stats,
                     });
  }];
  [command commit];
  return rund::AccelCheck{true, "ok"};
#else
  (void)adapter;
  (void)command_buffer;
  (void)completion;
  (void)user;
  return rund::AccelCheck{false, "accel_metal_unavailable"};
#endif
}

} // namespace rund::node::accel::detail

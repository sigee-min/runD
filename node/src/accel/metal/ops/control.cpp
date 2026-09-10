#include "internal.hpp"

#include "../buffer/owner.hpp"
#include "../stats.hpp"

#include <node/accel/buffer.hpp>

namespace rund::node::accel::detail::metal_ops_detail {

rund::RuntimeStats Stats(const rund::AccelDevice &pick) {
  const MetalRuntimeStats stats = ReadMetalRuntimeStats(pick);
  return stats.runtime;
}

rund::node::accel::AccelMemoryStats
Memory(const rund::AccelDevice &pick) noexcept {
  const MetalMemoryStats memory = ReadMetalMemoryStats(pick);
  return rund::node::accel::AccelMemoryStats{
      .staging =
          rund::node::accel::AccelMemoryCounter{.current = memory.current,
                                                .peak = memory.peak,
                                                .cumulative = memory.cumulative,
                                                .reused = memory.reused}};
}

rund::AccelCheck
VirtualPipelineCapability(const rund::AccelDevice &pick) noexcept {
  const MetalAdapter *const adapter = MetalAdapterFromPick(pick);
  if (adapter != nullptr &&
      adapter->residency_quarantined.load(std::memory_order_acquire)) {
    return rund::AccelCheck{false, "compute_device_lost"};
  }
  return adapter != nullptr && adapter->residency_queue != nullptr
             ? rund::AccelCheck{true, "ok"}
             : rund::AccelCheck{false, "compute_backend_unsupported"};
}

bool InjectDeviceLostOnce(const rund::AccelDevice &pick,
                          const SubmitKind kind) noexcept {
  MetalAdapter *const adapter = MetalAdapterFromPick(pick);
  if (adapter == nullptr) {
    return false;
  }
  return adapter->device_loss_fault.arm(kind);
}

bool InjectDownloadFailureOnce(const rund::AccelDevice &pick) noexcept {
  MetalAdapter *const adapter = MetalAdapterFromPick(pick);
  if (adapter == nullptr) {
    return false;
  }
  adapter->fault_download_once.store(true, std::memory_order_relaxed);
  return true;
}

bool InjectHostReadUnavailableOnce(const rund::AccelDevice &pick) noexcept {
  MetalAdapter *const adapter = MetalAdapterFromPick(pick);
  if (adapter == nullptr) {
    return false;
  }
  adapter->fault_host_read_once.store(true, std::memory_order_relaxed);
  return true;
}

bool InjectHostWriteUnavailableOnce(const rund::AccelDevice &pick) noexcept {
  MetalAdapter *const adapter = MetalAdapterFromPick(pick);
  if (adapter == nullptr) {
    return false;
  }
  adapter->fault_host_write_once.store(true, std::memory_order_relaxed);
  return true;
}

bool InjectResidencyTerminalLossOnce(const rund::AccelDevice &pick) noexcept {
  MetalAdapter *const adapter = MetalAdapterFromPick(pick);
  if (adapter == nullptr ||
      adapter->residency_quarantined.load(std::memory_order_acquire)) {
    return false;
  }
  adapter->fault_residency_terminal_once.store(true, std::memory_order_release);
  return true;
}

bool InjectTraceUnavailableOnce(const rund::AccelDevice &pick) noexcept {
  MetalAdapter *const adapter = MetalAdapterFromPick(pick);
  if (adapter == nullptr) {
    return false;
  }
  adapter->fault_trace_unavailable_once.store(true, std::memory_order_relaxed);
  return true;
}

bool InjectTraceResolveDeviceLostOnce(const rund::AccelDevice &pick) noexcept {
  MetalAdapter *const adapter = MetalAdapterFromPick(pick);
  if (adapter == nullptr) {
    return false;
  }
  adapter->fault_trace_resolve_device_lost_once.store(
      true, std::memory_order_relaxed);
  return true;
}

} // namespace rund::node::accel::detail::metal_ops_detail

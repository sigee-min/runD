#include <accel/device.hpp>

#include "owner.hpp"
#include <rund/counter.hpp>

#include <cmath>
#include <limits>

namespace rund::node::accel::detail {

MetalMemoryStats ReadMetalMemoryStats(const rund::AccelDevice &pick) noexcept {
  MetalAdapter *const adapter = MetalAdapterFromPick(pick);
  if (adapter == nullptr) {
    return {};
  }
  std::lock_guard lock{adapter->mutex};
  return adapter->memory;
}

void RecordMetalHostToDeviceBytes(MetalAdapter &adapter,
                                  const rund::kernel::u64 bytes) {
  std::lock_guard<std::mutex> lock{adapter.mutex};
  ::rund::detail::counter::Accumulate(
      adapter.stats.runtime.run.transfer.host_to_device_bytes, bytes);
}

void RecordMetalDeviceToHostBytes(MetalAdapter &adapter,
                                  const rund::kernel::u64 bytes) {
  std::lock_guard<std::mutex> lock{adapter.mutex};
  ::rund::detail::counter::Accumulate(
      adapter.stats.runtime.run.transfer.device_to_host_bytes, bytes);
}

void RecordMetalDispatch(MetalAdapter &adapter) {
  RecordMetalDispatches(adapter, 1u);
}

void RecordMetalDispatches(MetalAdapter &adapter,
                           const rund::kernel::u64 count) {
  if (count == 0u) {
    return;
  }
  std::lock_guard<std::mutex> lock{adapter.mutex};
  ::rund::detail::counter::Accumulate(
      adapter.stats.runtime.run.work.dispatch_count, count);
}

void RecordMetalCommandSubmitWaitNs(MetalAdapter &adapter,
                                    const std::uint64_t elapsed_ns) {
  std::lock_guard<std::mutex> lock{adapter.mutex};
  ::rund::detail::counter::Accumulate(
      adapter.stats.runtime.run.work.command_submit_count, 1u);
  ::rund::detail::counter::Accumulate(
      adapter.stats.runtime.run.time.command_submit_wait_ns, elapsed_ns);
}

void RecordMetalCommandSubmit(MetalAdapter &adapter) {
  std::lock_guard<std::mutex> lock{adapter.mutex};
  ::rund::detail::counter::Accumulate(
      adapter.stats.runtime.run.work.command_submit_count, 1u);
}

void RecordMetalCommandSubmitWaitOnlyNs(MetalAdapter &adapter,
                                        const std::uint64_t elapsed_ns) {
  std::lock_guard<std::mutex> lock{adapter.mutex};
  ::rund::detail::counter::Accumulate(
      adapter.stats.runtime.run.time.command_submit_wait_ns, elapsed_ns);
}

std::uint64_t BeginMetalResidencyCommand(MetalAdapter &adapter) noexcept {
  const std::uint64_t active = adapter.residency_command_active.fetch_add(
                                   1u, std::memory_order_acq_rel) +
                               1u;
  std::uint64_t peak =
      adapter.residency_command_peak.load(std::memory_order_acquire);
  while (peak < active && !adapter.residency_command_peak.compare_exchange_weak(
                              peak, active, std::memory_order_acq_rel,
                              std::memory_order_acquire)) {
  }
  return active;
}

bool EndMetalResidencyCommand(MetalAdapter &adapter) noexcept {
  const std::uint64_t active =
      adapter.residency_command_active.fetch_sub(1u, std::memory_order_acq_rel);
  if (active != 0u) {
    return true;
  }
  adapter.residency_command_active.store(0u, std::memory_order_release);
  return false;
}

std::uint64_t RecordMetalComputeKernelSeconds(MetalAdapter &adapter,
                                              const double start_seconds,
                                              const double end_seconds) {
  if (!std::isfinite(start_seconds) || !std::isfinite(end_seconds) ||
      end_seconds <= start_seconds) {
    return 0u;
  }
  const long double elapsed_ns = (static_cast<long double>(end_seconds) -
                                  static_cast<long double>(start_seconds)) *
                                 1000000000.0L;
  if (elapsed_ns <= 0.0L) {
    return 0u;
  }
  const auto ns_limit =
      static_cast<long double>(std::numeric_limits<std::uint64_t>::max());
  const std::uint64_t ns = elapsed_ns >= ns_limit
                               ? std::numeric_limits<std::uint64_t>::max()
                               : static_cast<std::uint64_t>(elapsed_ns);
  std::lock_guard<std::mutex> lock{adapter.mutex};
  ::rund::detail::counter::Accumulate(
      adapter.stats.runtime.run.time.accel_kernel_ns, ns);
  ::rund::detail::counter::Accumulate(
      adapter.stats.runtime.run.time.accel_timestamp_count, 1u);
  adapter.stats.runtime.run.time.accel_timestamp_source =
      "metal_command_buffer_compute_time";
  return ns;
}

void RecordMetalDispatchTrace(MetalAdapter &adapter,
                              const std::uint64_t elapsed_ns,
                              const std::uint64_t sample_count) {
  std::lock_guard<std::mutex> lock{adapter.mutex};
  ::rund::detail::counter::Accumulate(
      adapter.stats.runtime.run.time.accel_kernel_ns, elapsed_ns);
  ::rund::detail::counter::Accumulate(
      adapter.stats.runtime.run.time.accel_timestamp_count, sample_count);
  adapter.stats.runtime.run.time.accel_timestamp_source =
      "metal_dispatch_counter_timestamp_calibrated";
}

} // namespace rund::node::accel::detail

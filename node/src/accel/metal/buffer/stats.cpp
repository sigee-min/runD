#include <accel/device.hpp>

#include <rund/counter.hpp>
#include "owner.hpp"

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
  ::rund::detail::counter::Accumulate(adapter.stats.host_to_device_bytes,
                                      bytes);
}

void RecordMetalDeviceToHostBytes(MetalAdapter &adapter,
                                  const rund::kernel::u64 bytes) {
  std::lock_guard<std::mutex> lock{adapter.mutex};
  ::rund::detail::counter::Accumulate(adapter.stats.device_to_host_bytes,
                                      bytes);
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
  ::rund::detail::counter::Accumulate(adapter.stats.dispatch_count, count);
}

void RecordMetalCommandSubmitWaitNs(MetalAdapter &adapter,
                                    const std::uint64_t elapsed_ns) {
  std::lock_guard<std::mutex> lock{adapter.mutex};
  ::rund::detail::counter::Accumulate(adapter.stats.command_submit_count, 1u);
  ::rund::detail::counter::Accumulate(adapter.stats.command_submit_wait_ns,
                                      elapsed_ns);
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
  ::rund::detail::counter::Accumulate(adapter.stats.accel_kernel_ns, ns);
  ::rund::detail::counter::Accumulate(adapter.stats.accel_timestamp_count, 1u);
  adapter.stats.accel_timestamp_source = "metal_command_buffer_compute_time";
  return ns;
}

} // namespace rund::node::accel::detail

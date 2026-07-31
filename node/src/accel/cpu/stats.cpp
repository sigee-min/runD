#include <accel/device.hpp>
#include <accel/runtime.hpp>

#include "buffer.hpp"
#include <rund/counter.hpp>

namespace rund::node::accel::detail {

rund::RuntimeStats ReadCpuRuntimeStats(const rund::AccelDevice &pick) {
  CpuAdapter *const adapter = CpuAdapterFromPick(pick);
  if (adapter == nullptr) {
    return rund::RuntimeStats{.ok = false,
                              .reason = "accel_buffer_backend_unavailable"};
  }
  std::lock_guard<std::mutex> lock{adapter->mutex};
  return rund::RuntimeStats{
      .dispatch_count = adapter->dispatch_count,
      .reset_command_count = adapter->reset_command_count,
      .reset_bytes = adapter->reset_bytes,
      .buffer_allocation_count = adapter->buffer_allocation_count,
      .host_to_device_bytes = adapter->host_to_device_bytes,
      .device_to_host_bytes = adapter->device_to_host_bytes,
      .ok = true,
      .reason = "ok",
  };
}

void ResetCpuRuntimeStats(const rund::AccelDevice &pick) {
  CpuAdapter *const adapter = CpuAdapterFromPick(pick);
  if (adapter == nullptr) {
    return;
  }
  std::lock_guard<std::mutex> lock{adapter->mutex};
  adapter->dispatch_count = 0u;
  adapter->reset_command_count = 0u;
  adapter->reset_bytes = 0u;
  adapter->buffer_allocation_count = 0u;
  adapter->host_to_device_bytes = 0u;
  adapter->device_to_host_bytes = 0u;
}

void RecordCpuDispatches(CpuAdapter &adapter, const std::uint64_t count) {
  if (count == 0u) {
    return;
  }
  std::lock_guard<std::mutex> lock{adapter.mutex};
  ::rund::detail::counter::Accumulate(adapter.dispatch_count, count);
}

} // namespace rund::node::accel::detail

#pragma once

#include <rund/compute/backend.hpp>
#include <rund/compute/compile.hpp>
#include <rund/compute/device/pipeline/memory.hpp>
#include <rund/compute/status.hpp>
#include <rund/storage.hpp>

#include <accel/context/value.hpp>
#include <accel/device.hpp>
#include <node/runtime/backend.hpp>

#include <atomic>
#include <cstddef>
#include <cstdlib>
#include <memory>
#include <mutex>
#include <variant>

namespace rund::compute {
struct DeviceInfo;
}

namespace rund::compute::detail {

struct DeviceOps;
class CompileService;
namespace residency {
class Registry;
}

struct AllocationMeter final {
  AllocationMeter() noexcept = default;
  AllocationMeter(const AllocationMeter &) = delete;
  AllocationMeter &operator=(const AllocationMeter &) = delete;

  std::uint64_t current{};
  std::uint64_t peak{};
  std::uint64_t cumulative{};
  std::uint64_t reused{};
};

struct TrafficMeter final {
  TrafficMeter() noexcept = default;
  TrafficMeter(const TrafficMeter &) = delete;
  TrafficMeter &operator=(const TrafficMeter &) = delete;

  std::atomic<std::uint64_t> peak{};
  std::atomic<std::uint64_t> cumulative{};
};

static_assert(std::atomic<std::uint64_t>::is_always_lock_free,
              "warm Device traffic evidence requires lock-free uint64 atomics");

struct DeviceMemory final {
  // Serializes the allocate/check/record transaction for the paired logical
  // and committed allocation meters. Traffic is monotonic atomic state in the
  // same owner, so warm transfers never acquire this gate. This remains
  // independent of conservative Pipeline admission; no hidden capacity
  // authority exists.
  std::mutex gate;
  AllocationMeter logical;
  AllocationMeter host;
  AllocationMeter device;
  TrafficMeter transfer;
};

struct DeviceClaims final {
  std::mutex gate;
};

struct CpuDeviceState final {
  kernel::CpuCaps caps;
  node::BackendSelection workers;
};

struct AccelDeviceState final {
  rund::AccelDevice pick;
  rund::AccelContext context;
};

struct DeviceState final {
  Backend backend{Backend::Cpu};
  std::variant<CpuDeviceState, AccelDeviceState> storage;
  const DeviceOps *ops{};
  // Device identity is constructed once at open. Profile snapshots retain this
  // immutable owner instead of rebuilding owning strings on the warm path.
  std::shared_ptr<const DeviceInfo> info;
  // Immutable platform fact captured once while the Device opens. Planning
  // consumes this value and never re-queries the host page size.
  std::uint64_t host_page_bytes{};
  // Sole aggregate admission authority for all Pipelines on this Device.
  storage::Budget pipeline_memory_budget{};
  DeviceMemory memory;
  std::shared_ptr<CompileService> compile_owner;
  std::weak_ptr<CompileService> compile;
  Compile compile_resources{.workers = 0u, .capacity = 0u};
  std::shared_ptr<DeviceClaims> claims{std::make_shared<DeviceClaims>()};
  // Sole mutable page-to-frame authority for every virtual execution on this
  // Device. Physical backends consume leases; they never own a second cache.
  std::shared_ptr<residency::Registry> residency;
};

[[nodiscard]] Status
initialize_device_state(DeviceState &state,
                        DevicePipelineMemoryLimit pipeline_memory) noexcept;

void record_transfer(DeviceState &device, std::uint64_t bytes) noexcept;

[[nodiscard]] inline CpuDeviceState *cpu_device(DeviceState &device) noexcept {
  return std::get_if<CpuDeviceState>(&device.storage);
}
[[nodiscard]] inline const CpuDeviceState *
cpu_device(const DeviceState &device) noexcept {
  return std::get_if<CpuDeviceState>(&device.storage);
}
[[nodiscard]] inline AccelDeviceState *
accel_device(DeviceState &device) noexcept {
  return std::get_if<AccelDeviceState>(&device.storage);
}
[[nodiscard]] inline const AccelDeviceState *
accel_device(const DeviceState &device) noexcept {
  return std::get_if<AccelDeviceState>(&device.storage);
}

} // namespace rund::compute::detail

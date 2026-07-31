#pragma once

#include <rund/compute/abi/device.hpp>

#include <accel/context/buffer.hpp>
#include <accel/context/value.hpp>
#include <accel/device.hpp>
#include <node/accel/context.hpp>
#include <node/runtime/backend.hpp>

#include <atomic>
#include <cstddef>
#include <cstdlib>
#include <memory>
#include <mutex>
#include <variant>

namespace rund::compute::detail {

struct DeviceOps;
class CompileService;

struct MemoryMeter final {
  MemoryMeter() noexcept = default;
  MemoryMeter(const MemoryMeter &) = delete;
  MemoryMeter &operator=(const MemoryMeter &) = delete;
  MemoryMeter(MemoryMeter &&other) noexcept
      : current(other.current.load(std::memory_order_relaxed)),
        peak(other.peak.load(std::memory_order_relaxed)),
        cumulative(other.cumulative.load(std::memory_order_relaxed)),
        reused(other.reused.load(std::memory_order_relaxed)),
        budget(other.budget.load(std::memory_order_relaxed)) {}
  MemoryMeter &operator=(MemoryMeter &&other) noexcept {
    current.store(other.current.load(std::memory_order_relaxed),
                  std::memory_order_relaxed);
    peak.store(other.peak.load(std::memory_order_relaxed),
               std::memory_order_relaxed);
    cumulative.store(other.cumulative.load(std::memory_order_relaxed),
                     std::memory_order_relaxed);
    reused.store(other.reused.load(std::memory_order_relaxed),
                 std::memory_order_relaxed);
    budget.store(other.budget.load(std::memory_order_relaxed),
                 std::memory_order_relaxed);
    return *this;
  }

  std::atomic<std::uint64_t> current{};
  std::atomic<std::uint64_t> peak{};
  std::atomic<std::uint64_t> cumulative{};
  std::atomic<std::uint64_t> reused{};
  std::atomic<std::uint64_t> budget{};
};

struct DeviceMemory final {
  // Cold allocation/admission is serialized per Device. Pipeline preparation
  // holds this recursively while its sealed plan is materialized, so another
  // buffer or Pipeline cannot consume the capacity proved by the preflight.
  std::recursive_mutex gate;
  MemoryMeter host;
  MemoryMeter device;
  MemoryMeter transfer;
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
  DeviceMemory memory;
  std::shared_ptr<CompileService> compile_owner;
  std::weak_ptr<CompileService> compile;
  Compile compile_resources{.workers = 0u, .capacity = 0u};
  std::shared_ptr<DeviceClaims> claims{std::make_shared<DeviceClaims>()};
};

struct AlignedDelete final {
  void operator()(std::byte *const data) const noexcept { std::free(data); }
};

struct CpuBufferState final {
  std::unique_ptr<std::byte, AlignedDelete> data;
  std::size_t bytes{};
};

struct AccelBufferState final {
  rund::AccelBuffer buffer;
};

struct BufferState final {
  ~BufferState();

  std::shared_ptr<DeviceState> device;
  std::variant<CpuBufferState, AccelBufferState> storage;
  Type type{Type::I32};
  std::size_t count{};
  std::size_t bytes{};
  std::size_t physical_bytes{};
  std::uint32_t readers{};
  bool writer{};
  bool poisoned{};
  std::uint64_t generation{};
};

void record_buffer(DeviceState &device, std::uint64_t bytes,
                   bool reused = false) noexcept;
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
[[nodiscard]] inline CpuBufferState *cpu_buffer(BufferState &buffer) noexcept {
  return std::get_if<CpuBufferState>(&buffer.storage);
}
[[nodiscard]] inline const CpuBufferState *
cpu_buffer(const BufferState &buffer) noexcept {
  return std::get_if<CpuBufferState>(&buffer.storage);
}
[[nodiscard]] inline AccelBufferState *
accel_buffer(BufferState &buffer) noexcept {
  return std::get_if<AccelBufferState>(&buffer.storage);
}
[[nodiscard]] inline const AccelBufferState *
accel_buffer(const BufferState &buffer) noexcept {
  return std::get_if<AccelBufferState>(&buffer.storage);
}

} // namespace rund::compute::detail

#include "backend.hpp"
#include "buffer/local.hpp"
#include "device/state.hpp"
#include "size.hpp"
#include "status.hpp"
#include "type.hpp"
#include <rund/counter.hpp>

#include <cstring>
#include <memory>
#include <span>
#include <utility>

namespace rund::compute::detail {
namespace {

enum class BufferInitialization : unsigned char {
  Zeroed,
  FullOverwrite,
};

void raise_peak(MemoryMeter &meter, const std::uint64_t value) noexcept {
  std::uint64_t peak = meter.peak.load(std::memory_order_relaxed);
  while (peak < value && !meter.peak.compare_exchange_weak(
                             peak, value, std::memory_order_relaxed)) {
  }
}

[[nodiscard]] std::uint64_t accumulate(std::atomic<std::uint64_t> &counter,
                                       const std::uint64_t delta) noexcept {
  std::uint64_t current = counter.load(std::memory_order_relaxed);
  while (true) {
    const std::uint64_t next =
        ::rund::detail::counter::SaturatingAdd(current, delta);
    if (counter.compare_exchange_weak(current, next,
                                      std::memory_order_relaxed)) {
      return next;
    }
  }
}

void release(std::atomic<std::uint64_t> &counter,
             const std::uint64_t value) noexcept {
  std::uint64_t current = counter.load(std::memory_order_relaxed);
  while (true) {
    const std::uint64_t next =
        ::rund::detail::counter::Remaining(current, value);
    if (counter.compare_exchange_weak(current, next,
                                      std::memory_order_relaxed)) {
      return;
    }
  }
}

MemoryMeter &buffer_meter(DeviceState &device) noexcept {
  return device.backend == Backend::Cpu ? device.memory.host
                                        : device.memory.device;
}

[[nodiscard]] Result<std::shared_ptr<BufferState>>
make_buffer_impl(const std::shared_ptr<DeviceState> &device, const Type type,
                 const std::size_t count,
                 const BufferInitialization initialization) {
  if (device == nullptr) {
    return Result<std::shared_ptr<BufferState>>::fail(Reason::DeviceInvalid);
  }
  std::lock_guard memory_lock{device->memory.allocation_gate};
  const std::size_t bytes = type_bytes(type);
  std::size_t byte_count = 0u;
  if (bytes == 0u || !size::multiply(count, bytes, byte_count)) {
    return Result<std::shared_ptr<BufferState>>::fail(Reason::BufferCapacity);
  }

  try {
    auto buffer = std::make_shared<BufferState>();
    buffer->device = device;
    buffer->type = type;
    buffer->count = count;
    buffer->bytes = byte_count;
    if (count == 0u) {
      buffer->storage.emplace<CpuBufferState>();
      return Result<std::shared_ptr<BufferState>>::success(std::move(buffer));
    }
    if (device->backend == Backend::Cpu) {
      void *raw = nullptr;
      if (posix_memalign(&raw, 64u, byte_count) != 0 || raw == nullptr) {
        return Result<std::shared_ptr<BufferState>>::fail(
            Reason::BufferCapacity);
      }
      if (initialization == BufferInitialization::Zeroed) {
        std::memset(raw, 0, byte_count);
      }
      buffer->storage.emplace<CpuBufferState>(CpuBufferState{
          .data = std::unique_ptr<std::byte, AlignedDelete>(
              static_cast<std::byte *>(raw)),
          .bytes = byte_count,
      });
      buffer->physical_bytes = byte_count;
      record_buffer(*device, byte_count);
      return Result<std::shared_ptr<BufferState>>::success(std::move(buffer));
    }
    if (device->ops == nullptr || device->ops->allocate == nullptr) {
      return Result<std::shared_ptr<BufferState>>::fail(Reason::DeviceInvalid);
    }
    const Status allocated =
        device->ops->allocate(*device, *buffer, bytes, count,
                              initialization == BufferInitialization::Zeroed);
    if (!allocated) {
      return Result<std::shared_ptr<BufferState>>::fail(allocated.reason());
    }
    const AccelBufferState *const stored = accel_buffer(*buffer);
    const bool reused =
        stored != nullptr && stored->buffer.buffer.storage_reused;
    record_buffer(*device, buffer->physical_bytes, reused);
    return Result<std::shared_ptr<BufferState>>::success(std::move(buffer));
  } catch (const std::bad_alloc &) {
    return Result<std::shared_ptr<BufferState>>::fail(Reason::BufferCapacity);
  }
}

} // namespace

void record_buffer(DeviceState &device, const std::uint64_t bytes,
                   const bool reused) noexcept {
  MemoryMeter &meter = buffer_meter(device);
  const std::uint64_t current = accumulate(meter.current, bytes);
  (void)accumulate(meter.cumulative, bytes);
  if (reused) {
    (void)accumulate(meter.reused, bytes);
  }
  raise_peak(meter, current);
}

void record_transfer(DeviceState &device, const std::uint64_t bytes) noexcept {
  MemoryMeter &meter = device.memory.transfer;
  (void)accumulate(meter.cumulative, bytes);
  raise_peak(meter, bytes);
}

BufferState::~BufferState() {
  if (physical_bytes != 0u && device != nullptr) {
    release(buffer_meter(*device).current, physical_bytes);
  }
}

std::size_t buffer_size(const std::shared_ptr<BufferState> &state) noexcept {
  return state == nullptr ? 0u : state->count;
}

Result<std::shared_ptr<BufferState>>
make_buffer(const std::shared_ptr<DeviceState> &device, const Type type,
            const std::size_t count) {
  return make_buffer_impl(device, type, count, BufferInitialization::Zeroed);
}

Result<std::shared_ptr<BufferState>>
make_input_binding_buffer(const std::shared_ptr<DeviceState> &device,
                          const Type type, const std::size_t count) {
  return make_buffer_impl(device, type, count,
                          BufferInitialization::FullOverwrite);
}

Result<std::shared_ptr<BufferState>>
make_workspace_buffer(const std::shared_ptr<DeviceState> &device,
                      const std::size_t count) {
  return make_buffer_impl(device, Type::U32, count,
                          BufferInitialization::FullOverwrite);
}

Result<std::shared_ptr<BufferState>>
upload_raw(const std::shared_ptr<DeviceState> &device, const HostView input) {
  if (input.data == nullptr && input.count != 0u) {
    return Result<std::shared_ptr<BufferState>>::fail(Reason::ShapeMismatch);
  }
  auto result = make_input_binding_buffer(device, input.type, input.count);
  if (!result) {
    return result;
  }
  if (input.count == 0u) {
    return result;
  }
  const std::size_t bytes = input.count * type_bytes(input.type);
  if (device->backend == Backend::Cpu) {
    CpuBufferState *const buffer = cpu_buffer(*result.value());
    if (buffer == nullptr || buffer->bytes != bytes) {
      return Result<std::shared_ptr<BufferState>>::fail(
          Reason::TransferInvalid);
    }
    std::memcpy(buffer->data.get(), input.data, bytes);
    record_transfer(*device, bytes);
    return result;
  }
  if (device->ops == nullptr || device->ops->upload == nullptr) {
    return Result<std::shared_ptr<BufferState>>::fail(Reason::TransferInvalid);
  }
  const Status uploaded =
      device->ops->upload(*device, *result.value(), input.data, bytes);
  if (uploaded) {
    record_transfer(*device, bytes);
  }
  return uploaded
             ? result
             : Result<std::shared_ptr<BufferState>>::fail(uploaded.reason());
}

} // namespace rund::compute::detail

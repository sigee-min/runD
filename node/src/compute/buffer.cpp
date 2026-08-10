#include "backend.hpp"
#include "buffer/local.hpp"
#include "device/state.hpp"
#include "size.hpp"
#include "status.hpp"
#include "type.hpp"
#include <rund/counter.hpp>

#include <algorithm>
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

AllocationMeter &committed_buffer_meter(DeviceState &device) noexcept {
  return device.backend == Backend::Cpu ? device.memory.host
                                        : device.memory.device;
}

void record_allocation(AllocationMeter &meter, const std::uint64_t bytes,
                       const bool reused) noexcept {
  ::rund::detail::counter::Accumulate(meter.current, bytes);
  ::rund::detail::counter::Accumulate(meter.cumulative, bytes);
  if (reused) {
    ::rund::detail::counter::Accumulate(meter.reused, bytes);
  }
  meter.peak = std::max(meter.peak, meter.current);
}

void record_buffer(DeviceState &device, const std::uint64_t logical_bytes,
                   const std::uint64_t committed_bytes,
                   const bool reused = false) noexcept {
  record_allocation(device.memory.logical, logical_bytes, reused);
  record_allocation(committed_buffer_meter(device), committed_bytes, reused);
}

[[nodiscard]] Result<std::shared_ptr<BufferState>>
make_buffer_impl(const std::shared_ptr<DeviceState> &device, const Type type,
                 const std::size_t count,
                 const BufferInitialization initialization,
                 const std::uint64_t exact_storage_bytes = 0u) {
  if (device == nullptr) {
    return Result<std::shared_ptr<BufferState>>::fail(Reason::DeviceInvalid);
  }
  std::lock_guard memory_lock{device->memory.gate};
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
      record_buffer(*device, buffer->bytes, buffer->physical_bytes);
      return Result<std::shared_ptr<BufferState>>::success(std::move(buffer));
    }
    if (device->ops == nullptr || device->ops->allocate == nullptr) {
      return Result<std::shared_ptr<BufferState>>::fail(Reason::DeviceInvalid);
    }
    const Status allocated = device->ops->allocate(
        *device, *buffer, bytes, count,
        initialization == BufferInitialization::Zeroed, exact_storage_bytes);
    if (!allocated) {
      return Result<std::shared_ptr<BufferState>>::fail(allocated.reason());
    }
    if (buffer->physical_bytes < buffer->bytes) {
      return Result<std::shared_ptr<BufferState>>::fail(Reason::BufferCapacity);
    }
    const AccelBufferState *const stored = accel_buffer(*buffer);
    const bool reused =
        stored != nullptr && stored->buffer.buffer.storage_reused;
    record_buffer(*device, buffer->bytes, buffer->physical_bytes, reused);
    return Result<std::shared_ptr<BufferState>>::success(std::move(buffer));
  } catch (const std::bad_alloc &) {
    return Result<std::shared_ptr<BufferState>>::fail(Reason::BufferCapacity);
  }
}

} // namespace

void record_transfer(DeviceState &device, const std::uint64_t bytes) noexcept {
  TrafficMeter &meter = device.memory.transfer;
  std::uint64_t cumulative = meter.cumulative.load(std::memory_order_relaxed);
  for (;;) {
    const std::uint64_t next =
        ::rund::detail::counter::SaturatingAdd(cumulative, bytes);
    if (meter.cumulative.compare_exchange_weak(cumulative, next,
                                               std::memory_order_release,
                                               std::memory_order_relaxed)) {
      break;
    }
  }
  std::uint64_t peak = meter.peak.load(std::memory_order_relaxed);
  while (peak < bytes && !meter.peak.compare_exchange_weak(
                             peak, bytes, std::memory_order_release,
                             std::memory_order_relaxed)) {
  }
}

BufferState::~BufferState() {
  if ((bytes != 0u || physical_bytes != 0u) && device != nullptr) {
    std::lock_guard lock{device->memory.gate};
    ::rund::detail::counter::Release(device->memory.logical.current, bytes);
    ::rund::detail::counter::Release(committed_buffer_meter(*device).current,
                                     physical_bytes);
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
make_planned_input_binding_buffer(const std::shared_ptr<DeviceState> &device,
                                  const Type type, const std::size_t count,
                                  const std::uint64_t exact_storage_bytes) {
  return make_buffer_impl(device, type, count,
                          BufferInitialization::FullOverwrite,
                          exact_storage_bytes);
}

Result<std::shared_ptr<BufferState>>
make_workspace_buffer(const std::shared_ptr<DeviceState> &device,
                      const std::size_t count) {
  return make_buffer_impl(device, Type::U32, count,
                          BufferInitialization::FullOverwrite);
}

Result<std::shared_ptr<BufferState>>
make_planned_workspace_buffer(const std::shared_ptr<DeviceState> &device,
                              const std::size_t count) {
  std::size_t logical_bytes = 0u;
  if (device == nullptr ||
      !size::multiply(count, sizeof(std::uint32_t), logical_bytes)) {
    return Result<std::shared_ptr<BufferState>>::fail(Reason::PipelineCapacity);
  }
  const auto committed = planned_buffer_storage_bytes(*device, logical_bytes);
  return committed
             ? make_buffer_impl(device, Type::U32, count,
                                BufferInitialization::FullOverwrite, *committed)
             : Result<std::shared_ptr<BufferState>>::fail(committed.reason());
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
  const UploadResult uploaded =
      device->ops->upload(*device, *result.value(), input.data, bytes);
  if (uploaded.status) {
    record_transfer(*device, bytes);
  }
  return uploaded.status ? result
                         : Result<std::shared_ptr<BufferState>>::fail(
                               uploaded.status.reason());
}

} // namespace rund::compute::detail

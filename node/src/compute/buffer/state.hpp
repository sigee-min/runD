#pragma once

#include <rund/compute/abi/state.hpp>
#include <rund/compute/fixed.hpp>
#include <accel/context/buffer.hpp>

#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <memory>
#include <variant>

namespace rund::compute::detail {

struct AlignedDelete final {
  void operator()(std::byte *const data) const noexcept { std::free(data); }
};

struct CpuBufferState final {
  // Physical VSM extents may expose multiple typed semantic views over one
  // aligned byte owner.  The allocation-accounting BufferState remains the
  // root owner; aliases share this pointer and never publish a second meter.
  std::shared_ptr<std::byte> data;
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
  // Backend-published retained storage charge. This is not an OS/device
  // physical-residency observation.
  std::size_t physical_bytes{};
  // Non-null only for a semantic view over an existing physical allocation.
  // The root BufferState owns backend allocation accounting and native
  // lifetime; a view owns no second bytes and keeps that root alive here.
  std::shared_ptr<BufferState> physical_owner;
  // True only after make_buffer_impl publishes both Device allocation meters.
  // A rejected backend allocation may have populated the fields above, but it
  // owns no public accounting and its destructor must not release any meter.
  bool memory_accounted{};
  std::uint32_t readers{};
  bool writer{};
  bool poisoned{};
  std::uint64_t generation{};
};

[[nodiscard]] inline const BufferState *
physical_buffer_owner(const BufferState &buffer) noexcept {
  const BufferState *owner = &buffer;
  while (owner->physical_owner != nullptr) {
    owner = owner->physical_owner.get();
  }
  return owner;
}

[[nodiscard]] inline bool
same_physical_buffer(const BufferState &left,
                     const BufferState &right) noexcept {
  return physical_buffer_owner(left) == physical_buffer_owner(right);
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

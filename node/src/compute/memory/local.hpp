#pragma once

#include "../../accel/kernel/memory.hpp"
#include "../../array.hpp"
#include "../device/state.hpp"

#include <kernel/program/compute/retention.hpp>
#include <rund/compute/stats.hpp>
#include <rund/counter.hpp>

#include <algorithm>
#include <cstdint>
#include <memory>
#include <span>
#include <vector>

namespace rund::compute::detail {

struct JobState;

// Caller holds JobState::gate. The memory owner performs the projection while
// the Profile owner decides the observation epoch and public result.
[[nodiscard]] MemoryStats job_memory_locked(const JobState &state) noexcept;

[[nodiscard]] inline MemoryCounter
fixed_memory(const std::uint64_t bytes,
             const std::uint64_t reused = 0u) noexcept {
  return MemoryCounter{.current = bytes,
                       .peak = bytes,
                       .cumulative = bytes,
                       .reused = reused,
                       .budget = bytes};
}

[[nodiscard]] inline MemoryCounter
allocation_meter_memory(const AllocationMeter &meter) noexcept {
  return MemoryCounter{
      .current = meter.current,
      .peak = meter.peak,
      .cumulative = meter.cumulative,
      .reused = meter.reused,
  };
}

[[nodiscard]] inline MemoryCounter
traffic_meter_memory(const TrafficMeter &meter) noexcept {
  return MemoryCounter{
      .peak = meter.peak,
      .cumulative = meter.cumulative,
  };
}

inline void merge_memory(MemoryCounter &total,
                         const MemoryCounter part) noexcept {
  total.current =
      ::rund::detail::counter::SaturatingAdd(total.current, part.current);
  total.peak = ::rund::detail::counter::SaturatingAdd(total.peak, part.peak);
  total.cumulative =
      ::rund::detail::counter::SaturatingAdd(total.cumulative, part.cumulative);
  total.reused =
      ::rund::detail::counter::SaturatingAdd(total.reused, part.reused);
  total.budget =
      ::rund::detail::counter::SaturatingAdd(total.budget, part.budget);
}

inline void merge_memory(MemoryStats &total, const MemoryStats &part) noexcept {
  merge_memory(total.host, part.host);
  merge_memory(total.device, part.device);
  merge_memory(total.frame, part.frame);
  merge_memory(total.tile, part.tile);
  merge_memory(total.resident, part.resident);
  merge_memory(total.staging, part.staging);
  merge_memory(total.transfer, part.transfer);
}

template <class T>
[[nodiscard]] inline std::uint64_t
vector_memory(const std::vector<T> &values) noexcept {
  return kernel::compute_retained_detail::VectorCapacityBytes(values);
}

template <class T>
[[nodiscard]] inline std::uint64_t
vector_memory(const ::rund::node::detail::PreparedArray<T> &values) noexcept {
  return values.owned_bytes();
}

struct BufferMemory final {
  std::uint64_t resident{};
  std::uint64_t physical{};
  std::uint64_t reused{};
};

[[nodiscard]] inline BufferMemory
measure_buffer(const std::shared_ptr<BufferState> &buffer) noexcept {
  if (buffer == nullptr) {
    return {};
  }
  const AccelBufferState *const accel = accel_buffer(*buffer);
  const std::uint64_t physical = buffer->physical_bytes;
  return BufferMemory{
      .resident = buffer->bytes,
      .physical = physical,
      .reused = accel != nullptr && accel->buffer.buffer.storage_reused
                    ? physical
                    : 0u,
  };
}

inline void add_buffer_memory(BufferMemory &total,
                              const BufferMemory memory) noexcept {
  total.resident =
      ::rund::detail::counter::SaturatingAdd(total.resident, memory.resident);
  total.physical =
      ::rund::detail::counter::SaturatingAdd(total.physical, memory.physical);
  total.reused =
      ::rund::detail::counter::SaturatingAdd(total.reused, memory.reused);
}

template <class... Ranges>
[[nodiscard]] inline BufferMemory
measure_buffers(const Ranges &...ranges) noexcept {
  BufferMemory total{};
  const auto measure = [&total](const auto &buffers) noexcept {
    for (const auto &buffer : buffers) {
      add_buffer_memory(total, measure_buffer(buffer));
    }
  };
  (measure(ranges), ...);
  return total;
}

[[nodiscard]] inline Backend
memory_backend(const DeviceState &device) noexcept {
  return device.backend;
}

inline void set_physical(MemoryStats &stats, const std::uint64_t bytes,
                         const std::uint64_t reused = 0u) noexcept {
  if (stats.backend == Backend::Cpu) {
    stats.host = fixed_memory(
        ::rund::detail::counter::SaturatingAdd(stats.host.current, bytes),
        ::rund::detail::counter::SaturatingAdd(stats.host.reused, reused));
  } else {
    stats.device = fixed_memory(bytes, reused);
  }
}

class SnapshotWriter final {
public:
  SnapshotWriter(const MemoryStats summary,
                 const std::span<MemoryEntry> entries) noexcept
      : entries_(entries) {
    snapshot_.summary = summary;
  }

  void set_summary(const MemoryStats summary) noexcept {
    snapshot_.summary = summary;
  }

  void add(const MemoryCategory category, const MemoryUse use,
           const std::uint32_t index, const MemoryCounter bytes) noexcept {
    if (snapshot_.written < entries_.size()) {
      entries_[snapshot_.written++] = MemoryEntry{
          .category = category, .use = use, .index = index, .bytes = bytes};
    }
    ++snapshot_.total;
  }

  [[nodiscard]] MemorySnapshot finish() noexcept { return snapshot_; }

private:
  std::span<MemoryEntry> entries_;
  MemorySnapshot snapshot_{};
};

} // namespace rund::compute::detail

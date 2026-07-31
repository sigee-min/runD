#pragma once

#include <rund/session/scheduler.hpp>

#include <array>
#include <cstddef>
#include <cstdint>
#include <span>

#if defined(__unix__) || defined(__APPLE__)
#include <sys/uio.h>
#endif

namespace rund::node::nativeio {

#if defined(__unix__) || defined(__APPLE__)
using NativeSlice = ::iovec;
#else
struct NativeSlice final {
  void *iov_base = nullptr;
  std::size_t iov_len = 0u;
};
#endif

inline constexpr std::size_t VectoredCapacity =
    ::rund::SchedulerConfig{}.net_iov_capacity;

struct VectoredBatch final {
  std::array<NativeSlice, VectoredCapacity> native{};
  std::size_t count = 0u;
  bool valid = false;
  std::uint64_t admitted_bytes = 0u;
};

template <typename Slice>
[[nodiscard]] VectoredBatch
PrepareSlices(const std::span<const Slice> slices,
              const std::size_t slice_capacity,
              const std::size_t byte_capacity) noexcept {
  if (slices.empty() || slices.size() > slice_capacity ||
      slices.size() > VectoredCapacity) {
    return {};
  }

  VectoredBatch batch{};
  std::size_t bytes = 0u;
  for (std::size_t index = 0u; index < slices.size(); ++index) {
    const Slice slice = slices[index];
    if ((slice.data == nullptr && slice.size != 0u) ||
        slice.size > byte_capacity || bytes > byte_capacity - slice.size) {
      return {};
    }
    batch.native[index] = NativeSlice{
        .iov_base = const_cast<void *>(static_cast<const void *>(slice.data)),
        .iov_len = slice.size};
    bytes += slice.size;
  }
  batch.count = slices.size();
  batch.valid = true;
  batch.admitted_bytes = static_cast<std::uint64_t>(bytes);
  return batch;
}

} // namespace rund::node::nativeio

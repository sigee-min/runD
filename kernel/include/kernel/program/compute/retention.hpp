#pragma once

#include <kernel/program/compute/model.hpp>

#include <cstddef>
#include <functional>
#include <limits>
#include <memory>
#include <string>
#include <vector>

namespace rund::kernel::compute_retained_detail {

[[nodiscard]] constexpr u64 Add(const u64 left, const u64 right) noexcept {
  return right > std::numeric_limits<u64>::max() - left
             ? std::numeric_limits<u64>::max()
             : left + right;
}

[[nodiscard]] constexpr u64 CapacityBytes(const std::size_t capacity,
                                          const std::size_t width) noexcept {
  if (width != 0u && capacity > std::numeric_limits<u64>::max() / width) {
    return std::numeric_limits<u64>::max();
  }
  return static_cast<u64>(capacity) * static_cast<u64>(width);
}

template <class T>
[[nodiscard]] inline u64
VectorCapacityBytes(const std::vector<T> &values) noexcept {
  return CapacityBytes(values.capacity(), sizeof(T));
}

[[nodiscard]] inline u64
StringExternalStorageBytes(const std::string &value) noexcept {
  if (value.capacity() == 0u) {
    return 0u;
  }
  const void *const object_begin =
      static_cast<const void *>(std::addressof(value));
  const void *const object_end = static_cast<const void *>(
      reinterpret_cast<const unsigned char *>(object_begin) + sizeof(value));
  const void *const data = static_cast<const void *>(value.data());
  const std::less<const void *> before{};
  const bool object_local =
      !before(data, object_begin) && before(data, object_end);
  if (object_local) {
    return 0u;
  }
  return Add(CapacityBytes(value.capacity(), sizeof(char)), sizeof(char));
}

} // namespace rund::kernel::compute_retained_detail

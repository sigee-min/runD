#pragma once

#include <kernel/core/checked.hpp>

#include <cstddef>
#include <cstdint>
#include <limits>
#include <new>
#include <span>
#include <type_traits>

namespace rund::compute::detail {

struct CpuArenaSegment final {
  std::uint64_t offset_bytes{};
  std::uint64_t size_bytes{};
  std::size_t count{};
  std::size_t element_bytes{};
  std::size_t alignment{};

  [[nodiscard]] constexpr bool empty() const noexcept { return count == 0u; }
  [[nodiscard]] constexpr bool
  operator==(const CpuArenaSegment &) const noexcept = default;
};

// Canonical append-only typed layout. Every consumer appends segments in a
// fixed semantic order, then seals the complete mapping exactly once.
struct CpuArenaLayout final {
  std::uint64_t extent_bytes{};
  std::uint64_t committed_bytes{};
  std::uint64_t page_bytes{};
  std::uint64_t maximum_alignment{1u};
  bool sealed{};

  [[nodiscard]] constexpr bool
  operator==(const CpuArenaLayout &) const noexcept = default;
};

[[nodiscard]] bool append_cpu_arena_segment(CpuArenaLayout &layout,
                                            std::size_t count,
                                            std::size_t element_bytes,
                                            std::size_t alignment,
                                            CpuArenaSegment &segment) noexcept;

template <class T>
[[nodiscard]] bool append_cpu_arena_segment(CpuArenaLayout &layout,
                                            const std::size_t count,
                                            CpuArenaSegment &segment) noexcept {
  static_assert(!std::is_void_v<T>);
  return append_cpu_arena_segment(layout, count, sizeof(T), alignof(T),
                                  segment);
}

[[nodiscard]] bool seal_cpu_arena_layout(CpuArenaLayout &layout,
                                         std::uint64_t page_bytes) noexcept;

class CpuArenaMapping final {
public:
  CpuArenaMapping() noexcept = default;
  CpuArenaMapping(const CpuArenaMapping &) = delete;
  CpuArenaMapping &operator=(const CpuArenaMapping &) = delete;
  CpuArenaMapping(CpuArenaMapping &&other) noexcept;
  CpuArenaMapping &operator=(CpuArenaMapping &&other) noexcept;
  ~CpuArenaMapping();

  [[nodiscard]] bool allocate(const CpuArenaLayout &layout) noexcept;
  void release() noexcept;

  [[nodiscard]] bool valid() const noexcept {
    return committed_bytes_ == 0u || data_ != nullptr;
  }
  [[nodiscard]] std::uint64_t extent_bytes() const noexcept {
    return extent_bytes_;
  }
  [[nodiscard]] std::uint64_t committed_bytes() const noexcept {
    return committed_bytes_;
  }

  template <class T>
  [[nodiscard]] std::span<T>
  construct(const CpuArenaSegment &segment) noexcept {
    if (!contains<T>(segment)) {
      return {};
    }
    if (segment.count == 0u) {
      return {};
    }
    T *const values = reinterpret_cast<T *>(data_ + segment.offset_bytes);
    for (std::size_t index = 0u; index < segment.count; ++index) {
      ::new (static_cast<void *>(values + index)) T{};
    }
    return {values, segment.count};
  }

  [[nodiscard]] std::span<std::byte>
  bytes(const CpuArenaSegment &segment) noexcept {
    if (segment.element_bytes != 1u || segment.count != segment.size_bytes ||
        segment.offset_bytes > extent_bytes_ ||
        segment.size_bytes > extent_bytes_ - segment.offset_bytes ||
        (segment.count != 0u &&
         (data_ == nullptr || segment.alignment == 0u ||
          segment.offset_bytes % segment.alignment != 0u))) {
      return {};
    }
    return segment.count == 0u
               ? std::span<std::byte>{}
               : std::span<std::byte>{data_ + segment.offset_bytes,
                                      segment.count};
  }

  template <class T> void destroy(const std::span<T> values) noexcept {
    for (std::size_t index = values.size(); index != 0u; --index) {
      values[index - 1u].~T();
    }
  }

private:
  template <class T>
  [[nodiscard]] bool contains(const CpuArenaSegment &segment) const noexcept {
    if (segment.count == 0u) {
      return segment.size_bytes == 0u && segment.element_bytes == sizeof(T) &&
             segment.alignment == alignof(T);
    }
    if (data_ == nullptr || segment.element_bytes != sizeof(T) ||
        segment.alignment != alignof(T) ||
        segment.offset_bytes % alignof(T) != 0u ||
        segment.offset_bytes > extent_bytes_ ||
        segment.size_bytes > extent_bytes_ - segment.offset_bytes ||
        segment.count > std::numeric_limits<std::uint64_t>::max() / sizeof(T)) {
      return false;
    }
    return segment.size_bytes ==
           static_cast<std::uint64_t>(segment.count) * sizeof(T);
  }

  std::byte *data_{};
  std::uint64_t extent_bytes_{};
  std::uint64_t committed_bytes_{};
};

} // namespace rund::compute::detail

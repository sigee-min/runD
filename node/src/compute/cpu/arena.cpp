#if defined(__linux__) && !defined(_DEFAULT_SOURCE)
#define _DEFAULT_SOURCE 1
#endif

#include "arena.hpp"

#include <sys/mman.h>

#include <algorithm>
#include <cstring>
#include <limits>
#include <utility>

namespace rund::compute::detail {

bool append_cpu_arena_segment(CpuArenaLayout &layout, const std::size_t count,
                              const std::size_t element_bytes,
                              const std::size_t alignment,
                              CpuArenaSegment &segment) noexcept {
  if (layout.sealed || element_bytes == 0u || alignment == 0u ||
      (alignment & (alignment - 1u)) != 0u) {
    return false;
  }
  std::uint64_t offset = 0u;
  std::uint64_t size = 0u;
  std::uint64_t next = 0u;
  if (!kernel::checked::align_up(layout.extent_bytes, alignment, offset) ||
      !kernel::checked::mul(static_cast<std::uint64_t>(count),
                            static_cast<std::uint64_t>(element_bytes), size) ||
      !kernel::checked::add(offset, size, next)) {
    return false;
  }
  segment = CpuArenaSegment{
      .offset_bytes = offset,
      .size_bytes = size,
      .count = count,
      .element_bytes = element_bytes,
      .alignment = alignment,
  };
  layout.extent_bytes = next;
  layout.maximum_alignment =
      std::max(layout.maximum_alignment, static_cast<std::uint64_t>(alignment));
  return true;
}

bool seal_cpu_arena_layout(CpuArenaLayout &layout,
                           const std::uint64_t page_bytes) noexcept {
  if (layout.sealed || page_bytes == 0u ||
      (page_bytes & (page_bytes - 1u)) != 0u ||
      page_bytes < layout.maximum_alignment ||
      page_bytes % layout.maximum_alignment != 0u) {
    return false;
  }
  std::uint64_t committed = 0u;
  if (!kernel::checked::align_up(layout.extent_bytes, page_bytes, committed) ||
      committed > std::numeric_limits<std::size_t>::max()) {
    return false;
  }
  layout.page_bytes = page_bytes;
  layout.committed_bytes = committed;
  layout.sealed = true;
  return true;
}

CpuArenaMapping::CpuArenaMapping(CpuArenaMapping &&other) noexcept
    : data_(std::exchange(other.data_, nullptr)),
      extent_bytes_(std::exchange(other.extent_bytes_, 0u)),
      committed_bytes_(std::exchange(other.committed_bytes_, 0u)) {}

CpuArenaMapping &CpuArenaMapping::operator=(CpuArenaMapping &&other) noexcept {
  if (this == &other) {
    return *this;
  }
  release();
  data_ = std::exchange(other.data_, nullptr);
  extent_bytes_ = std::exchange(other.extent_bytes_, 0u);
  committed_bytes_ = std::exchange(other.committed_bytes_, 0u);
  return *this;
}

CpuArenaMapping::~CpuArenaMapping() { release(); }

bool CpuArenaMapping::allocate(const CpuArenaLayout &layout) noexcept {
  if (data_ != nullptr || extent_bytes_ != 0u || committed_bytes_ != 0u ||
      !layout.sealed || layout.extent_bytes > layout.committed_bytes ||
      layout.committed_bytes > std::numeric_limits<std::size_t>::max() ||
      (layout.committed_bytes != 0u &&
       (layout.page_bytes == 0u ||
        layout.committed_bytes % layout.page_bytes != 0u))) {
    return false;
  }
  if (layout.committed_bytes == 0u) {
    return layout.extent_bytes == 0u;
  }
#if defined(MAP_ANONYMOUS)
  constexpr int anonymous = MAP_ANONYMOUS;
#elif defined(MAP_ANON)
  constexpr int anonymous = MAP_ANON;
#else
  return false;
#endif
  void *const mapped =
      ::mmap(nullptr, static_cast<std::size_t>(layout.committed_bytes),
             PROT_READ | PROT_WRITE, MAP_PRIVATE | anonymous, -1, 0);
  if (mapped == MAP_FAILED) {
    return false;
  }
  data_ = static_cast<std::byte *>(mapped);
  extent_bytes_ = layout.extent_bytes;
  committed_bytes_ = layout.committed_bytes;
  // Eagerly first-touch the complete mapping so the committed contract is a
  // cold preparation fact rather than a warm-path demand-fault surprise.
  std::memset(data_, 0, static_cast<std::size_t>(committed_bytes_));
  return true;
}

void CpuArenaMapping::release() noexcept {
  if (data_ != nullptr) {
    (void)::munmap(data_, static_cast<std::size_t>(committed_bytes_));
  }
  data_ = nullptr;
  extent_bytes_ = 0u;
  committed_bytes_ = 0u;
}

} // namespace rund::compute::detail

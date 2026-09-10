#pragma once

#include <cstddef>
#include <cstdint>
#include <limits>

namespace rund::compute::detail::residency {

enum class Failure : std::uint8_t { None, Invalid, Capacity, Infeasible };

struct PageKey final {
  std::uint32_t resource{};
  std::uint64_t page{};

  [[nodiscard]] constexpr bool
  operator==(const PageKey &) const noexcept = default;
  [[nodiscard]] constexpr bool operator<(const PageKey &other) const noexcept {
    return resource < other.resource ||
           (resource == other.resource && page < other.page);
  }
};

enum class Access : std::uint8_t { Read, Write, ReadWrite };

[[nodiscard]] constexpr bool reads(const Access access) noexcept {
  return access != Access::Write;
}
[[nodiscard]] constexpr bool writes(const Access access) noexcept {
  return access != Access::Read;
}

inline constexpr std::uint64_t NeverUse =
    std::numeric_limits<std::uint64_t>::max();

struct DirtyRange final {
  std::uint64_t offset{};
  std::uint64_t bytes{};
  [[nodiscard]] constexpr bool
  operator==(const DirtyRange &) const noexcept = default;
};

struct PinInterval final {
  std::uint64_t first_epoch{};
  std::uint64_t last_epoch{};
  [[nodiscard]] constexpr bool
  operator==(const PinInterval &) const noexcept = default;
};

struct PageDemand final {
  PageKey key{};
  Access access{Access::Read};
  DirtyRange dirty{};
  [[nodiscard]] constexpr bool
  operator==(const PageDemand &) const noexcept = default;
};

struct PageUse final {
  PageKey key{};
  Access access{Access::Read};
  DirtyRange dirty{};
  std::uint64_t next_use{NeverUse};
  PinInterval pin{};
  std::uint64_t prefetch_epoch{};
  std::uint64_t ready_epoch{};
  [[nodiscard]] constexpr bool
  operator==(const PageUse &) const noexcept = default;
};

enum class TransitionKind : std::uint8_t {
  Writeback,
  Migrate,
  Discard,
  Unmap,
  Fetch,
  Map,
};

struct Epoch final {
  std::uint64_t ordinal{};
  std::uint32_t node{};
  std::uint32_t tile{};
  std::size_t first_use{};
  std::size_t use_count{};
};

struct PageRun final {
  std::uint64_t first_page{};
  std::uint64_t page_count{};
  PinInterval pin{};
  std::uint64_t prefetch_epoch{};
  std::uint64_t ready_epoch{};
};

struct Identity final {
  std::uint64_t hi{};
  std::uint64_t lo{};
  [[nodiscard]] explicit operator bool() const noexcept {
    return hi != 0u || lo != 0u;
  }
  [[nodiscard]] bool operator==(const Identity &) const noexcept = default;
};

} // namespace rund::compute::detail::residency

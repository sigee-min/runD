#pragma once

#include "../../../pipeline/residency/model.hpp"

#include <cstdint>

namespace rund::compute::detail::residency {

enum class CacheDomain : std::uint8_t { Backing, Transient };

struct CacheKey final {
  std::uint64_t backing{};
  std::uint64_t version{};
  // Zero identifies the complete backing. Nonzero distinguishes a boundary
  // materialization whose clamped values depend on an active prefix extent.
  std::uint64_t extent{};
  std::uint64_t materialization_hi{};
  std::uint64_t materialization_lo{};
  std::uint64_t page{};
  // Transient identifies a VSM-owned intermediate with no backing write
  // target. It shares the Authority table and replacement boundary, but must
  // be consumed or explicitly discarded rather than counted as writeback.
  CacheDomain domain{CacheDomain::Backing};

  [[nodiscard]] constexpr bool
  operator==(const CacheKey &) const noexcept = default;
  [[nodiscard]] constexpr bool operator<(const CacheKey &other) const noexcept {
    if (domain != other.domain) {
      return domain < other.domain;
    }
    if (backing != other.backing) {
      return backing < other.backing;
    }
    if (version != other.version) {
      return version < other.version;
    }
    if (extent != other.extent) {
      return extent < other.extent;
    }
    if (materialization_hi != other.materialization_hi) {
      return materialization_hi < other.materialization_hi;
    }
    if (materialization_lo != other.materialization_lo) {
      return materialization_lo < other.materialization_lo;
    }
    return page < other.page;
  }
};

// Exact authoritative byte interval in the CacheKey content domain. For a
// Backing key it is absolute in that backing; for a Transient key it is the
// immutable intermediate-materialization range. Empty means clean. Migration
// carries this extent unchanged, and only Backing keys may reach writeback.
struct DirtyExtent final {
  std::uint64_t offset{};
  std::uint64_t bytes{};

  [[nodiscard]] constexpr bool empty() const noexcept { return bytes == 0u; }
  [[nodiscard]] constexpr bool
  operator==(const DirtyExtent &) const noexcept = default;
};

struct CacheUse final {
  CacheKey key{};
  Access access{Access::Read};
  std::uint64_t next_use{};
  DirtyExtent dirty{};
  // Generic Graph admission carries the planner's relative epoch and the
  // last epoch through which this materialization must survive. NeverUse in
  // both fields marks a non-Graph Stream admission.
  std::uint64_t epoch{NeverUse};
  std::uint64_t retain_until{NeverUse};
};

enum class FrameState : std::uint8_t {
  Empty,
  Mapping,
  Resident,
  Pinned,
  Dirty,
  Writeback,
};

enum class FrameTier : std::uint8_t { Host, Device };

// Tier describes physical placement; role describes the only legal consumer
// of a registered region. Keeping both in the one frame table prevents a Host
// input-cache lease from accidentally admitting Host output bytes (or vice
// versa) while still avoiding a tier-local page table.
enum class FrameRole : std::uint8_t { Input, Intermediate, Output };

struct FrameRegion final {
  FrameTier tier{FrameTier::Device};
  FrameRole role{FrameRole::Input};
  std::uint32_t first{};
  std::uint32_t count{};

  [[nodiscard]] constexpr bool
  operator==(const FrameRegion &) const noexcept = default;
};

} // namespace rund::compute::detail::residency

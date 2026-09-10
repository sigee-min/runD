#pragma once

#include "../../../pipeline/residency/model.hpp"
#include "cache_model.hpp"

#include <array>
#include <cstddef>
#include <cstdint>
#include <span>

namespace rund::compute::detail::residency {

// Immutable binding from one canonical Graph resource to its CacheKey domain
// and byte geometry. PageUse supplies page, access, future-use, pin, dirty,
// ready, and prefetch facts; runtime callers may not reconstruct those facts
// into a second policy schedule.
struct GraphMaterialization final {
  std::uint32_t resource{};
  CacheKey key{};
  std::uint64_t page_bytes{};
  std::uint64_t page_count{};
  std::uint64_t boundary_extent{};
};

// The sole symbolic-resource -> runtime CacheKey projection. In particular,
// only the right-boundary page carries the active invocation extent; complete
// pages remain reusable across active prefixes.
[[nodiscard]] bool
project_graph_cache_key(const GraphMaterialization &materialization,
                        PageKey page, CacheKey &key) noexcept;

struct GraphPortRequest final {
  std::uint16_t program_port{};
  std::size_t first_use{};
  std::size_t use_count{};
  GraphMaterialization materialization{};
  // Exact executable K-prefix. Prepared commands address only this region.
  FrameRegion region{};
  // Complete compatible arena coordinates. Cross-K borrowers may find a hit
  // in either canonical C-sized bank, but relocation must land in `region`
  // before the prepared stage becomes Computing.
  std::array<FrameRegion, 2u> cache_regions{};
  std::size_t cache_region_count{};
  std::span<const GraphPageRemap> remaps{};
};

struct GraphLeasePort final {
  std::uint16_t program_port{};
  Access access{Access::Read};
  std::uint32_t resource{};
  FrameRegion region{};
  std::array<FrameRegion, 2u> cache_regions{};
  std::size_t cache_region_count{};
  std::size_t first_binding{};
  std::size_t binding_count{};
  std::size_t first_remap{};
  std::size_t remap_count{};
};

// One dependency-ordered physical page copy. A cycle is lowered through an
// evictable arena frame, so callers execute this span strictly in order.
struct GraphRelocation final {
  std::uint16_t port{};
  std::uint32_t source_frame{};
  std::uint32_t target_frame{};
  std::uint64_t bytes{};
};

struct GraphFreezeRequest final {
  static constexpr std::size_t RegionCapacity = 4u;

  GraphMaterialization materialization{};
  std::span<const FrameRegion> regions;
};
} // namespace rund::compute::detail::residency

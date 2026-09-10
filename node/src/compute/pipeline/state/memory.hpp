#pragma once

#include <rund/compute/stats.hpp>

#include <array>
#include <cstdint>
#include <vector>

namespace rund::compute::detail {

// Ordinals borrow the immutable execution routes; no additional Job lifetime
// owner or address index survives preparation.
struct PipelineMemoryOwner final {
  std::uint32_t ordinal{};
  std::uint32_t cpu_storage{~std::uint32_t{0u}};
  std::uint64_t metadata{};
};

static_assert(sizeof(PipelineMemoryOwner) == 16u);

// Only immutable ownership categories can be retained here. Frame, transfer,
// staging and native prepared counters have no slot in this sealed record.
struct PipelineSharedOwnership final {
  MemoryCounter host{};
  MemoryCounter device{};
  MemoryCounter tile{};
  MemoryCounter resident{};
};

struct PipelineMemoryInventory final {
  std::vector<PipelineMemoryOwner> jobs;
  // Inclusive and Pool-exclusive physical ownership projections. Dynamic
  // backend, transfer and frame counters are deliberately absent at sealing.
  std::array<PipelineSharedOwnership, 2u> shared;
  std::uint64_t metadata{};
  std::size_t cpu_storage_count{};
  std::uint64_t scratch_resident{};
  std::uint64_t scratch_physical{};
  std::uint64_t scratch_reused{};
  std::uint64_t referenced_resource_bytes{};
};

} // namespace rund::compute::detail

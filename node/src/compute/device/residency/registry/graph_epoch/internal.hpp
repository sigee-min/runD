#pragma once

#include "../credentials/cpu.hpp"
#include "../credentials/epoch.hpp"
#include "../graph.hpp"
#include "../model/frame.hpp"
#include "../model/lease.hpp"

#include <array>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <span>

namespace rund::compute::detail::residency {

class Authority;

namespace graph_epoch_detail {

inline constexpr std::size_t MaxUses =
    TiledGraphPortCapacity * PipelineLeafCapacity;
inline constexpr std::uint32_t NoFrame =
    std::numeric_limits<std::uint32_t>::max();
using Score = std::int64_t;
inline constexpr Score Forbidden = std::numeric_limits<Score>::max() / 8;

// One invocation's bounded planning record. It is created by the Authority
// entry point and consumed before the prepared LeaseSlot is published. It
// owns no physical rows, policy, or retained execution state.
struct AdmissionDraft final {
  std::size_t page_count{};
  std::size_t local_count{};
  std::size_t port_count{};
  bool has_remap{};
  std::array<CacheUse, MaxUses> projected{};
  std::array<std::uint32_t, MaxUses> existing_frames{};
  std::array<std::array<Score, PipelineLeafCapacity>, PipelineLeafCapacity>
      costs{};
  std::array<std::uint32_t, PipelineLeafCapacity> locals{};
  Score maximum_score{};
};

// Request/remap/materialization validation and PageUse projection. Region
// validation is a second read-only entry so admission can retain its exact
// state-precondition order before inspecting physical frames.
class Validation final {
public:
  [[nodiscard]] static AuthorityResult
  requests(std::span<const PageUse>, std::span<const GraphPortRequest>,
           std::size_t anchor_port, std::uint64_t epoch,
           AdmissionDraft &) noexcept;
  [[nodiscard]] static AuthorityResult
  regions(const Authority &, std::span<const GraphPortRequest>) noexcept;
};

// Existing-frame discovery, deterministic hit/replacement scoring, and the
// fixed rectangular Hungarian assignment. This phase only reads Authority
// rows and writes the bounded draft.
class Assignment final {
public:
  [[nodiscard]] static AuthorityResult run(const Authority &,
                                           std::span<const GraphPortRequest>,
                                           std::uint64_t,
                                           AdmissionDraft &) noexcept;
};

// Relocation/permutation planning and LeaseSlot/frame staging. The caller
// owns the Authority gate; failures use the existing undo journal to restore
// the sole physical table before returning the original failure class.
class Relocation final {
public:
  [[nodiscard]] static AuthorityResult
  run(Authority &, std::span<const PageUse>, std::span<const GraphPortRequest>,
      std::uint64_t, AdmissionDraft &, registry_model::LeaseSlot &) noexcept;
};

} // namespace graph_epoch_detail
} // namespace rund::compute::detail::residency

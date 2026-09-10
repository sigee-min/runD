#pragma once

#include "../backing.hpp"
#include "../cache.hpp"
#include "../projection.hpp"
#include "../../graph/reduce/timeline.hpp"

#include <array>
#include <cstddef>
#include <cstdint>
#include <memory>

namespace rund::compute::detail::virtual_run_overlap {

using TimelineInterval = graph_reduce::Interval;

// This is the value-only state passed between the direct phase owners. It
// contains one bounded epoch projection and its authenticated physical lease;
// preparation, native completion, and output publication mutate only their
// own fields. Timeline arithmetic is owned by graph_reduce::Timeline.
struct PreparedEpoch final {
  VirtualEpochProjection projection{};
  std::shared_ptr<PipelineState> pipeline;
  std::array<residency::CacheBinding, PipelineLeafCapacity * 2u> bindings{};
  std::array<residency::CacheTransition, PipelineLeafCapacity * 6u>
      transitions{};
  std::array<residency::CacheKey, PipelineLeafCapacity> keys{};
  std::array<residency::CacheTransition, PipelineLeafCapacity> drains{};
  std::array<const std::byte *, PipelineLeafCapacity> output_frames{};
  VirtualOutputReservation output{};
  graph_reduce::Timeline timeline{};
  std::size_t binding_count{};
  std::size_t transition_count{};
  std::size_t drain_count{};
  std::uint64_t token{};
  std::uint64_t drain_token{};
  std::uint64_t ordinal{};
  bool submitted{};
  bool stats_folded{};
  bool coherent_input{};
  bool coherent_output{};
};

struct PageOutTimeline final {
  TimelineInterval download{};
  TimelineInterval backing{};
};

[[nodiscard]] bool cycle_flight(const PreparedEpoch &,
                                const VirtualRunProjection &,
                                residency::cycle::Flight &) noexcept;

[[nodiscard]] residency::EpochLease input_lease(PreparedEpoch &) noexcept;
[[nodiscard]] residency::EpochLease execution_output_lease(
    PreparedEpoch &) noexcept;
[[nodiscard]] residency::EpochLease execution_lease(PreparedEpoch &) noexcept;

[[nodiscard]] Status fold_epoch(PreparedEpoch &, Stats &) noexcept;

} // namespace rund::compute::detail::virtual_run_overlap

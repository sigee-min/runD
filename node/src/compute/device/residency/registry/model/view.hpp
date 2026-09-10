#pragma once

#include "../../../../pipeline/residency/model.hpp"
#include "../graph.hpp"

#include <cstddef>
#include <cstdint>
#include <span>

namespace rund::compute::detail::residency::registry_model {

// Source-private all-or-nothing view transition. Graph supplies activation
// regions and freeze requests; Stream supplies stream regions. ViewOwner
// validates both phases under Authority's gate before applying either phase.
struct ViewCommitPlan final {
  enum class Kind : std::uint8_t { Stream, Graph };
  static constexpr std::size_t GraphRequestCapacity =
      TiledGraphResourceCapacity - 2u;

  Kind kind{Kind::Stream};
  std::span<const FrameRegion> activation_regions{};
  std::span<const FrameRegion> stream_regions{};
  const StreamPlan *stream{};
  CacheKey last{};
  const TiledGraphInvocation *graph{};
  std::span<const GraphFreezeRequest> requests{};
};

enum class ViewCommitState : std::uint8_t {
  Idle,
  Active,
  Quarantined
};

inline constexpr std::size_t GraphForecastQuarantineCapacity = 4u;

} // namespace rund::compute::detail::residency::registry_model

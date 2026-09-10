#include "../planner.hpp"

#include "../identity.hpp"
#include "graph/internal.hpp"

#include "../../../type.hpp"
#include "../../plan/contract.hpp"

#include <kernel/core/checked.hpp>

#include <algorithm>
#include <cstdint>
#include <limits>
#include <new>
#include <utility>

namespace rund::compute::detail::residency {

PlanResult PlanResidency(const TiledGraphPlanInput &source) noexcept {
  if (source.page_count == 0u || source.resources.empty() ||
      source.resources.size() > TiledGraphResourceCapacity ||
      source.stages.empty()) {
    return PlanResult{.failure = Failure::Invalid};
  }
  if (source.requested_frames > source.max_frames ||
      source.max_frames > std::numeric_limits<std::uint32_t>::max()) {
    return PlanResult{.failure = Failure::Capacity};
  }
  if (source.requested_frames == 0u) {
    return PlanResult{.failure = Failure::Infeasible};
  }
  const std::uint64_t frames =
      std::min(source.page_count, source.requested_frames);
  try {
    GraphPlanningState state{.input = source};
    state.frames = frames;
    Failure failure = initialize_graph(state);
    if (failure != Failure::None) {
      return PlanResult{.failure = failure};
    }
    failure = freeze_graph_liveness(state);
    if (failure != Failure::None) {
      return PlanResult{.failure = failure};
    }
    failure = assign_graph_physical_classes(state);
    if (failure != Failure::None) {
      return PlanResult{.failure = failure};
    }
    failure = seal_graph_dependencies(state);
    if (failure != Failure::None) {
      return PlanResult{.failure = failure};
    }
    const std::uint64_t batches =
        state.input.page_count / frames +
        static_cast<std::uint64_t>(state.input.page_count % frames != 0u);
    std::uint64_t epochs = 0u;
    if (batches > std::numeric_limits<std::uint32_t>::max() ||
        !kernel::checked::mul(batches, state.stages.size(), epochs) ||
        epochs == 0u || epochs == NeverUse) {
      return PlanResult{.failure = Failure::Capacity};
    }
    const Identity identity = IdentifyResidencyPlan(
        state.input.page_count, frames, state.input.prefetch_distance,
        state.resources, state.physical_classes, state.stages,
        state.input.graph_fingerprint_hi, state.input.graph_fingerprint_lo);
    TiledGraphPlan tiled{
        state.input.page_count,
        frames,
        state.input.prefetch_distance,
        std::move(state.resources),
        std::move(state.physical_classes),
        std::move(state.stages),
        state.input.graph_fingerprint_hi,
        state.input.graph_fingerprint_lo,
    };
    return PlanResult{
        .failure = Failure::None,
        .plan = ResidencyPlan{state.page_bytes, std::move(tiled), identity},
    };
  } catch (const std::bad_alloc &) {
    return PlanResult{.failure = Failure::Capacity};
  }
}

} // namespace rund::compute::detail::residency

#include "internal.hpp"

namespace rund::compute::detail::graph_reduce {

PrefetchController::PrefetchController(
    VirtualPipelineState &state, const std::span<VirtualBacking *const> inputs,
    const VirtualRunProjection &run, Stats &stats, residency::Pool &pool,
    const residency::TiledGraphPlan &graph, Wavefront &wavefront,
    const std::uint64_t batches) noexcept
    : state_(state), run_(run), stats_(stats), pool_(pool),
      authority_(pool.authority()), graph_(graph), wavefront_(wavefront),
      batches_(batches) {
  input_count_ = std::min(inputs.size(), inputs_.size());
  std::copy(inputs.begin(),
            inputs.begin() + static_cast<std::ptrdiff_t>(input_count_),
            inputs_.begin());
}

std::size_t
PrefetchController::lane_index(const std::uint64_t batch) const noexcept {
  return static_cast<std::size_t>(batch % lanes_.size());
}

PrefetchLane &PrefetchController::lane(const std::uint64_t batch) noexcept {
  return lanes_[lane_index(batch)];
}

const PrefetchLane &
PrefetchController::lane(const std::uint64_t batch) const noexcept {
  return lanes_[lane_index(batch)];
}

bool PrefetchController::quiescent() const noexcept {
  for (std::size_t index = 0u; index < lanes_.size(); ++index) {
    if (lanes_[index].pending || lanes_[index].forecast ||
        !pool_.prefetch[index].quiescent()) {
      return false;
    }
  }
  return true;
}

bool PrefetchController::workers_quiescent() const noexcept {
  for (std::size_t index = 0u; index < lanes_.size(); ++index) {
    if (!pool_.prefetch[index].quiescent()) {
      return false;
    }
  }
  return true;
}

bool PrefetchController::settle(PrefetchLane &lane) noexcept {
  if (!lane.forecast) {
    return true;
  }
  bool settled = authority_.graph_forecasts().release_graph_forecast(
      std::move(lane.forecast));
  if (!settled) {
    settled = authority_.graph_forecasts().abort_graph_forecast(lane.forecast);
  }
  return settled;
}

bool PrefetchController::pair_supported() const noexcept {
  return graph_wavefront_pair_eligible(state_, run_);
}

bool PrefetchController::has_pending(const std::uint64_t batch) const noexcept {
  for (const PrefetchLane &candidate : lanes_) {
    if (candidate.pending && candidate.batch == batch && candidate.stage != 0u &&
        candidate.stage + 1u < graph_.stages().size()) {
      return true;
    }
  }
  return false;
}

} // namespace rund::compute::detail::graph_reduce

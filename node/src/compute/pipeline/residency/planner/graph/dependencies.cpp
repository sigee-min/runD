#include "internal.hpp"

#include <algorithm>
#include <array>
#include <cstdint>
#include <limits>
#include <span>

namespace rund::compute::detail::residency {

namespace {

class Dependencies final {
public:
  [[nodiscard]] bool append(const std::size_t stage,
                            const TiledGraphDependencyPhase phase) noexcept {
    if (stage > std::numeric_limits<std::uint32_t>::max()) {
      return false;
    }
    const TiledGraphStageDependency value{
        .stage = static_cast<std::uint32_t>(stage), .phase = phase};
    const auto end = entries_.begin() + count_;
    const auto found = std::lower_bound(
        entries_.begin(), end, value,
        [](const auto edge, const auto candidate) {
          return edge.stage < candidate.stage;
        });
    if (found != end && found->stage == value.stage) {
      found->phase = std::max(found->phase, value.phase);
    } else {
      if (count_ == entries_.size()) {
        return false;
      }
      std::move_backward(found, end, end + 1u);
      *found = value;
      ++count_;
    }
    return true;
  }

  [[nodiscard]] std::span<const TiledGraphStageDependency> entries() const
      noexcept {
    return std::span{entries_}.first(count_);
  }

private:
  std::array<TiledGraphStageDependency, TiledGraphDependencyCapacity> entries_;
  std::size_t count_{};
};

} // namespace

Failure seal_graph_dependencies(GraphPlanningState &state) {
  // Seal the execution wavefront from the same resource/liveness authority
  // that produced PageUse. Runtime admission must consume these coordinates
  // directly; ordinal adjacency is not a Graph dependency.
  for (std::size_t stage_index = 0u; stage_index < state.stages.size();
       ++stage_index) {
    TiledGraphStage &stage = state.stages[stage_index];
    Dependencies same_batch;
    Dependencies prior_batch;
    for (const TiledGraphPort port : stage.ports) {
      const TiledGraphResource *const current =
          find_graph_resource(state, port.resource);
      if (current == nullptr) {
        return Failure::Invalid;
      }
      // A transient consumer depends on its exact producer. Distinct
      // consumers do not depend on one another and may form a wavefront.
      if (reads(port.access) &&
          current->persistence == ResourcePersistence::Transient) {
        if (current->producer_stage == NoGraphStage ||
            current->producer_stage >= stage_index ||
            !same_batch.append(current->producer_stage,
                               TiledGraphDependencyPhase::DispatchComplete)) {
          return Failure::Invalid;
        }
      }
      if (current->first_stage != stage_index) {
        continue;
      }
      const TiledGraphResource *previous_owner = nullptr;
      const TiledGraphResource *last_owner = current;
      bool first_owner = true;
      for (const TiledGraphResource &candidate : state.resources) {
        if (candidate.physical_id != current->physical_id) {
          continue;
        }
        if (candidate.first_stage < current->first_stage) {
          first_owner = false;
        }
        if (candidate.last_stage < current->first_stage &&
            (previous_owner == nullptr ||
             previous_owner->last_stage < candidate.last_stage)) {
          previous_owner = &candidate;
        }
        if (last_owner->last_stage < candidate.last_stage) {
          last_owner = &candidate;
        }
      }
      // Interval coloring aliases disjoint logical lifetimes. The first use
      // of the next logical owner waits only the immediately preceding
      // owner's exact last consumer, not every earlier stage.
      if (previous_owner != nullptr &&
          !same_batch.append(
              previous_owner->last_stage,
              previous_owner->kind == GraphResourceKind::ExternalOutput
                  ? TiledGraphDependencyPhase::ReleaseComplete
                  : TiledGraphDependencyPhase::DispatchComplete)) {
        return Failure::Capacity;
      }
      // Cyclic frame locals are reused by the next batch. Only the first
      // logical owner of this physical class waits the prior batch's exact
      // last consumer; this introduces no backing/Host-I/O dependency.
      if (first_owner &&
          !prior_batch.append(
              last_owner->last_stage,
              last_owner->kind == GraphResourceKind::ExternalOutput
                  ? TiledGraphDependencyPhase::ReleaseComplete
                  : TiledGraphDependencyPhase::DispatchComplete)) {
        return Failure::Capacity;
      }
    }
    const auto same = same_batch.entries();
    const auto prior = prior_batch.entries();
    if (same.size() > TiledGraphDependencyCapacity - prior.size()) {
      return Failure::Capacity;
    }
    stage.same_batch_predecessors.assign(same.begin(), same.end());
    stage.prior_batch_predecessors.assign(prior.begin(), prior.end());
  }
  return Failure::None;
}

} // namespace rund::compute::detail::residency

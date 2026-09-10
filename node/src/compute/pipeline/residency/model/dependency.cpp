#include "../model.hpp"

#include <kernel/core/checked.hpp>

#include <algorithm>

namespace rund::compute::detail::residency {

bool TiledGraphInvocation::predecessors(
    const std::uint64_t batch_index, const std::size_t stage_index,
    const std::span<TiledGraphDependency> storage,
    std::size_t &count) const noexcept {
  count = 0u;
  if (!valid() || batch_index >= batch_count() ||
      stage_index >= plan_->stages_.size()) {
    return false;
  }
  const TiledGraphStage &stage = plan_->stages_[stage_index];
  const std::size_t prior_count =
      batch_index == 0u ? 0u : stage.prior_batch_predecessors.size();
  if (stage.prior_batch_predecessors.size() > TiledGraphDependencyCapacity ||
      stage.same_batch_predecessors.size() >
          TiledGraphDependencyCapacity -
              stage.prior_batch_predecessors.size() ||
      storage.size() < prior_count + stage.same_batch_predecessors.size() ||
      !std::is_sorted(stage.same_batch_predecessors.begin(),
                      stage.same_batch_predecessors.end()) ||
      std::adjacent_find(stage.same_batch_predecessors.begin(),
                         stage.same_batch_predecessors.end()) !=
          stage.same_batch_predecessors.end() ||
      !std::is_sorted(stage.prior_batch_predecessors.begin(),
                      stage.prior_batch_predecessors.end()) ||
      std::adjacent_find(stage.prior_batch_predecessors.begin(),
                         stage.prior_batch_predecessors.end()) !=
          stage.prior_batch_predecessors.end()) {
    return false;
  }
  const auto append =
      [&](const std::uint64_t batch,
          const TiledGraphStageDependency predecessor) noexcept {
        std::uint64_t ordinal = 0u;
        if (predecessor.stage >= plan_->stage_count() ||
            !kernel::checked::mul(batch, plan_->stage_count(), ordinal) ||
            !kernel::checked::add(ordinal, predecessor.stage, ordinal)) {
          return false;
        }
        storage[count++] = TiledGraphDependency{.ordinal = ordinal,
                                                .batch = batch,
                                                .stage = predecessor.stage,
                                                .phase = predecessor.phase};
        return true;
      };
  if (batch_index != 0u) {
    for (const TiledGraphStageDependency predecessor :
         stage.prior_batch_predecessors) {
      if (!append(batch_index - 1u, predecessor)) {
        count = 0u;
        return false;
      }
    }
  }
  for (const TiledGraphStageDependency predecessor :
       stage.same_batch_predecessors) {
    if (predecessor.stage >= stage_index || !append(batch_index, predecessor)) {
      count = 0u;
      return false;
    }
  }
  return true;
}

} // namespace rund::compute::detail::residency

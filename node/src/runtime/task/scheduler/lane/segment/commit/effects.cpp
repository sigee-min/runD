#include <rund/task/stats/slots.hpp>

#include "../../../state/model/task.hpp"
#include "../../../state/segment.hpp"
#include "../../../state/storage.hpp"

#include <cstdlib>

namespace rund::node {

void Scheduler::CommitLaneOwnedSegmentEffects(
    const std::vector<LaneSegmentEffect> &lane_effects) noexcept {
  if (!lane_effects.empty()) {
    ++::rund::detail::task::Stat(
        state_->evidence.metrics,
        ::rund::detail::task::StatSlot::DeterministicEffectLogMergeBatches);
    ::rund::detail::task::Stat(
        state_->evidence.metrics,
        ::rund::detail::task::StatSlot::EffectEventsMerged) +=
        lane_effects.size();
  }
  LaneOwnedTerminalRange terminal_range{};
  for (const LaneSegmentEffect &effect : lane_effects) {
    TaskRecord *const record = effect.record;
    if (record == nullptr || record->id != effect.task_id) {
      continue;
    }
    if (effect.trapped) {
      CommitLaneOwnedPrimitiveTrapEffect(terminal_range, effect);
    }
    if (CommitLaneOwnedTerminalEffect(terminal_range, effect)) {
      continue;
    }
    if (effect.trapped) {
      std::abort();
    }
  }
  FlushLaneOwnedTerminalRange(terminal_range);
}

} // namespace rund::node

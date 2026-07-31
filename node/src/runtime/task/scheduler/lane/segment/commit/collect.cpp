#include "../../../state/segment.hpp"
#include "../../../state/storage.hpp"

#include <cstdlib>

namespace rund::node {

void Scheduler::CollectLaneOwnedSegmentEffects(
    const std::vector<LaneOwnedSegmentLane> &segments,
    std::vector<std::uint64_t> &executed_task_ids,
    std::vector<LaneSegmentEffect> &lane_effects) noexcept {
  for (const LaneOwnedSegmentLane &segment : segments) {
    if (!segment.submitted) {
      continue;
    }
    const std::size_t mark_count =
        std::min(segment.completed, segment.jobs.size());
    std::size_t effect_index = 0u;
    for (std::size_t offset = 0u; offset < mark_count; ++offset) {
      const LaneSegmentJob &job = segment.jobs[offset];
      executed_task_ids.push_back(job.task_id);
      if (effect_index < segment.effects.size() &&
          segment.effects[effect_index].canonical_index ==
              job.canonical_index) {
        lane_effects.push_back(segment.effects[effect_index]);
        ++effect_index;
      } else {
        lane_effects.push_back(LaneSegmentEffect{
            .record = job.record,
            .task_id = job.task_id,
            .logical_ticket = job.logical_ticket,
            .canonical_index = job.canonical_index,
            .terminal_kind = ::rund::detail::task::OperationKind::Complete,
            .code = ReasonCode::Ok,
            .terminal = true,
            .trapped = false,
            .trap_kind = ::rund::detail::task::OperationKind::PrimitiveTrap,
            .trap_code = ReasonCode::Ok,
        });
      }
    }
    if (effect_index != segment.effects.size()) {
      std::abort();
    }
  }
  std::sort(executed_task_ids.begin(), executed_task_ids.end());
  std::sort(
      lane_effects.begin(), lane_effects.end(),
      [](const LaneSegmentEffect &lhs, const LaneSegmentEffect &rhs) noexcept {
        if (lhs.canonical_index != rhs.canonical_index) {
          return lhs.canonical_index < rhs.canonical_index;
        }
        if (lhs.task_id != rhs.task_id) {
          return lhs.task_id < rhs.task_id;
        }
        if (lhs.logical_ticket != rhs.logical_ticket) {
          return lhs.logical_ticket < rhs.logical_ticket;
        }
        return lhs.terminal_kind < rhs.terminal_kind;
      });
}

} // namespace rund::node

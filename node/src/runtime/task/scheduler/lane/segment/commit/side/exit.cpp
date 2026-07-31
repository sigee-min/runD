#include <rund/task/stats/slots.hpp>

#include "../../../../state/model/task.hpp"
#include "../../../../state/storage.hpp"

namespace rund::node {

void Scheduler::MarkLaneOwnedSegmentSideExits(
    const std::vector<std::uint64_t> &task_ids,
    const std::vector<std::uint64_t> &executed_task_ids) noexcept {
  for (const std::uint64_t task_id : task_ids) {
    const bool was_executed = std::binary_search(
        executed_task_ids.begin(), executed_task_ids.end(), task_id);
    TaskRecord *const record = state_->Find(task_id);
    if (!was_executed && record != nullptr &&
        record->state == TaskState::Ready) {
      ++::rund::detail::task::Stat(state_->evidence.metrics,
                                   ::rund::detail::task::StatSlot::SideExits);
    }
  }
}

void Scheduler::RequeueLaneOwnedSegmentSideExits(
    const std::vector<std::uint64_t> &task_ids) noexcept {
  for (std::size_t reverse = task_ids.size(); reverse > 0u; --reverse) {
    const std::uint64_t id = task_ids[reverse - 1u];
    (void)RequeueReadyTask(id, 0u);
  }
}

} // namespace rund::node

#include "../state/model/task.hpp"
#include "../state/storage.hpp"

#include <rund/task/channel/access.hpp>

#include <algorithm>
#include <limits>
#include <vector>

namespace rund::detail::task {

ChannelDecision ChannelAccess::WakeChannelWaiters(
    std::vector<std::uint64_t> task_ids, const std::uint64_t channel_id,
    std::vector<std::uint64_t> *const failed_out) noexcept {
  if (failed_out != nullptr) {
    failed_out->clear();
  }
  node::Scheduler *const scheduler = node::Scheduler::Active();
  if (scheduler == nullptr) {
    if (failed_out != nullptr) {
      *failed_out = std::move(task_ids);
    }
    return ChannelDecision{
        .status = ::rund::task::Status::fail(ReasonCode::NodeRuntimeMissing)};
  }
  std::sort(task_ids.begin(), task_ids.end(),
            [scheduler](const std::uint64_t lhs, const std::uint64_t rhs) {
              const node::TaskRecord *const left = scheduler->state_->Find(lhs);
              const node::TaskRecord *const right =
                  scheduler->state_->Find(rhs);
              const std::uint64_t left_wait =
                  left != nullptr ? left->wait_id
                                  : std::numeric_limits<std::uint64_t>::max();
              const std::uint64_t right_wait =
                  right != nullptr ? right->wait_id
                                   : std::numeric_limits<std::uint64_t>::max();
              if (left_wait != right_wait) {
                return left_wait < right_wait;
              }
              return lhs < rhs;
            });
  ChannelDecision first_failure{};
  bool saw_failure = false;
  for (const std::uint64_t task_id : task_ids) {
    const ChannelDecision wake = scheduler->WakeChannel(task_id, channel_id);
    if (!wake.status) {
      if (!saw_failure) {
        first_failure = wake;
        saw_failure = true;
      }
      if (failed_out != nullptr) {
        failed_out->push_back(task_id);
      }
    }
  }
  if (saw_failure) {
    return first_failure;
  }
  return ChannelDecision{.status = ::rund::task::Status::success()};
}

} // namespace rund::detail::task

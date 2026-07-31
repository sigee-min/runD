#include "local.hpp"

namespace rund::node {
namespace {

enum class PendingJob : std::uint8_t {
  None,
  Regular,
  External,
  Direct,
  Mailbox,
};

void PreferEarlier(PendingJob candidate,
                   std::uint64_t ticket,
                   PendingJob* selected,
                   std::uint64_t* selected_ticket) noexcept {
  if (*selected == PendingJob::None || ticket == 0u ||
      (*selected_ticket != 0u && ticket < *selected_ticket)) {
    *selected = candidate;
    *selected_ticket = ticket;
  }
}

}  // namespace

bool LaneWorkerAccess::Claim(TaskLane &lane, LaneJobFrame &frame) noexcept {
  ResetFrame(frame);

  for (;;) {
    std::unique_lock<std::mutex> lock(lane.mutex);
    const auto ready_predicate = [&lane] {
      return lane.has_job || lane.stop || lane.external_wake_head != nullptr ||
             lane.work_head != nullptr || lane.direct_ready_head != nullptr ||
             lane.mailbox_state.load(std::memory_order_acquire) == 1u;
    };
    if (!ready_predicate()) {
      lane.ready_wait_requested = true;
      lane.ready.wait(lock, ready_predicate);
    }
    lane.ready_wait_requested = false;
    if (lane.stop) {
      return false;
    }

    if (lane.has_job && lane.segment_job_count != 0u) {
      LoadSegmentJob(lane, frame);
      return true;
    }

    if (lane.work_head != nullptr) {
      LoadWork(lane, frame);
      return true;
    }

    PendingJob selected = PendingJob::None;
    std::uint64_t selected_ticket = 0u;
    if (lane.has_job) {
      PreferEarlier(PendingJob::Regular, lane.commit_ticket, &selected,
                    &selected_ticket);
    }
    if (lane.external_wake_head != nullptr) {
      PreferEarlier(PendingJob::External,
                    lane.external_wake_head->wake_ticket, &selected,
                    &selected_ticket);
    }
    if (lane.direct_ready_head != nullptr) {
      PreferEarlier(PendingJob::Direct,
                    lane.direct_ready_head->wake_ticket, &selected,
                    &selected_ticket);
    }
    if (lane.mailbox_state.load(std::memory_order_acquire) == 1u) {
      PreferEarlier(PendingJob::Mailbox,
                    lane.mailbox_commit_ticket.load(std::memory_order_relaxed),
                    &selected, &selected_ticket);
    }

    switch (selected) {
      case PendingJob::Regular:
        LoadRegularJob(lane, frame);
        return true;
      case PendingJob::External:
        LoadExternalWakeJob(lane, frame);
        return true;
      case PendingJob::Direct:
        LoadDirectReadyJob(lane, frame);
        return true;
      case PendingJob::Mailbox: {
        std::uint8_t expected_mailbox = 1u;
        if (lane.mailbox_state.compare_exchange_strong(
                expected_mailbox, 2u, std::memory_order_acquire,
                std::memory_order_relaxed)) {
          LoadMailboxJob(lane, frame);
          lane.running = true;
          return true;
        }
        break;
      }
      case PendingJob::None:
        break;
    }
  }
}

bool ClaimLaneJob(TaskLane &lane, LaneJobFrame &frame) noexcept {
  return LaneWorkerAccess::Claim(lane, frame);
}

} // namespace rund::node

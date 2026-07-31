#include <rund/task/stats/slots.hpp>

#include "../state/model/context.hpp"
#include "../state/model/lane.hpp"
#include "../state/storage.hpp"
#include "../state/task/commit.hpp"

#include <cstdlib>
#include <limits>

namespace rund::node {

Scheduler::ControlCommitScope::ControlCommitScope(Scheduler &scheduler) noexcept
    : scheduler_(&scheduler) {
  if (active_scheduler_context != nullptr &&
      active_scheduler_context->scheduler == scheduler_) {
    return;
  }
  context_.scheduler = scheduler_;
  context_.scope_id = scheduler_->state_->identity.active_scope_id;
  context_.commit_ticket = scheduler_->IssueCommitTicket();
  context_.root_submit_recorded = true;
  context_.previous = active_scheduler_context;
  active_scheduler_context = &context_;
  scheduler_->EnsureCurrentCommit();
  installed_ = true;
}

Scheduler::ControlCommitScope::~ControlCommitScope() {
  if (!installed_) {
    return;
  }
  scheduler_->ReleaseQuantumCommit();
  active_scheduler_context = context_.previous;
}

std::uint64_t Scheduler::IssueCommitTicket() noexcept {
  return IssueCommitTickets(1u);
}

std::uint64_t Scheduler::IssueCommitTickets(
    const std::uint64_t logical_events) noexcept {
  if (logical_events == 0u) {
    return 0u;
  }
  std::lock_guard<std::mutex> lock(state_->batches.commit_mutex);
  const std::uint64_t first = state_->batches.next_commit_ticket_to_issue;
  if (first == 0u ||
      logical_events > std::numeric_limits<std::uint64_t>::max() - first) {
    std::abort();
  }
  state_->batches.next_commit_ticket_to_issue = first + logical_events;
  return first;
}

std::uint64_t Scheduler::IssueLaneCommitTickets(
    const std::uint64_t logical_events) noexcept {
  if (logical_events == 0u ||
      logical_events != state_->lanes.segment_commit_lanes.size()) {
    return 0u;
  }
  for (const std::uint32_t lane_index :
       state_->lanes.segment_commit_lanes) {
    if (lane_index >= state_->lanes.lanes.size()) {
      return 0u;
    }
  }

  std::lock_guard<std::mutex> lock(state_->batches.commit_mutex);
  if (state_->batches.lane_commit_active.load(std::memory_order_acquire) ||
      state_->batches.next_commit_ticket !=
          state_->batches.next_commit_ticket_to_issue) {
    return 0u;
  }
  const std::uint64_t first = state_->batches.next_commit_ticket_to_issue;
  if (first == 0u ||
      logical_events > std::numeric_limits<std::uint64_t>::max() - first) {
    return 0u;
  }
  const std::uint64_t end = first + logical_events;
  for (const auto &lane : state_->lanes.lanes) {
    lane->segment_completed_ticket.store(0u, std::memory_order_relaxed);
  }
  state_->batches.next_commit_ticket_to_issue = end;
  state_->batches.lane_commit_first.store(first, std::memory_order_relaxed);
  state_->batches.lane_commit_frontier.store(first,
                                              std::memory_order_relaxed);
  state_->batches.lane_commit_end.store(end, std::memory_order_relaxed);
  state_->batches.lane_commit_waiters.store(0u, std::memory_order_relaxed);
  state_->batches.lane_commit_active.store(true, std::memory_order_release);
  return first;
}

bool Scheduler::LaneCommitContains(const std::uint64_t ticket) const noexcept {
  if (!state_->batches.lane_commit_active.load(std::memory_order_acquire)) {
    return false;
  }
  const std::uint64_t first =
      state_->batches.lane_commit_first.load(std::memory_order_relaxed);
  const std::uint64_t end =
      state_->batches.lane_commit_end.load(std::memory_order_relaxed);
  return ticket >= first && ticket < end;
}

bool Scheduler::WaitLaneCommit(const std::uint64_t ticket) noexcept {
  std::uint64_t frontier =
      state_->batches.lane_commit_frontier.load(std::memory_order_acquire);
  const bool immediate = frontier == ticket;
  if (frontier > ticket) {
    std::abort();
  }
  if (!immediate) {
    state_->batches.lane_commit_waiters.fetch_add(1u,
                                                  std::memory_order_acq_rel);
    for (;;) {
      frontier =
          state_->batches.lane_commit_frontier.load(std::memory_order_acquire);
      if (frontier == ticket) {
        break;
      }
      if (frontier > ticket) {
        std::abort();
      }
      state_->batches.lane_commit_frontier.wait(frontier,
                                                std::memory_order_acquire);
    }
    state_->batches.lane_commit_waiters.fetch_sub(1u,
                                                  std::memory_order_acq_rel);
  }
  return immediate;
}

void Scheduler::AdvanceLaneCommit() noexcept {
  if (!state_->batches.lane_commit_active.load(std::memory_order_acquire)) {
    return;
  }
  const std::uint64_t first =
      state_->batches.lane_commit_first.load(std::memory_order_relaxed);
  const std::uint64_t end =
      state_->batches.lane_commit_end.load(std::memory_order_relaxed);
  std::uint64_t frontier =
      state_->batches.lane_commit_frontier.load(std::memory_order_acquire);

  while (frontier < end) {
    std::uint64_t candidate = frontier;
    while (candidate < end) {
      const std::size_t index = static_cast<std::size_t>(candidate - first);
      if (index >= state_->lanes.segment_commit_lanes.size()) {
        std::abort();
      }
      const std::uint32_t lane_index =
          state_->lanes.segment_commit_lanes[index];
      if (lane_index >= state_->lanes.lanes.size() ||
          state_->lanes.lanes[lane_index]->segment_completed_ticket.load(
              std::memory_order_acquire) < candidate) {
        break;
      }
      ++candidate;
    }
    if (candidate == frontier) {
      break;
    }
    if (state_->batches.lane_commit_frontier.compare_exchange_weak(
            frontier, candidate, std::memory_order_acq_rel,
            std::memory_order_acquire)) {
      frontier = candidate;
      if (state_->batches.lane_commit_waiters.load(
              std::memory_order_acquire) != 0u) {
        state_->batches.lane_commit_frontier.notify_all();
      }
    }
  }

  if (state_->batches.lane_commit_frontier.load(std::memory_order_acquire) !=
      end) {
    return;
  }
  bool active = true;
  if (!state_->batches.lane_commit_active.compare_exchange_strong(
          active, false, std::memory_order_acq_rel,
          std::memory_order_acquire)) {
    return;
  }
  {
    std::lock_guard<std::mutex> lock(state_->batches.commit_mutex);
    if (state_->batches.next_commit_ticket != first) {
      std::abort();
    }
    state_->batches.next_commit_ticket = end;
  }
  state_->batches.lane_commit_first.store(0u, std::memory_order_relaxed);
  state_->batches.lane_commit_end.store(0u, std::memory_order_relaxed);
  state_->batches.commit_cv.notify_all();
}

void Scheduler::CompleteLaneCommit(const std::uint64_t ticket) noexcept {
  if (!LaneCommitContains(ticket) || active_task_lane == nullptr) {
    std::abort();
  }
  active_task_lane->segment_completed_ticket.store(ticket,
                                                    std::memory_order_release);
  AdvanceLaneCommit();
}

void Scheduler::EnsureCurrentCommit() noexcept {
  SchedulerThreadContext *const context = active_scheduler_context;
  if (context == nullptr || context->scheduler != this) {
    return;
  }
  if (!context->commit_acquired && context->commit_ticket != 0u) {
    bool immediate = false;
    if (LaneCommitContains(context->commit_ticket)) {
      immediate = WaitLaneCommit(context->commit_ticket);
      context->commit_acquired = true;
    } else {
      std::unique_lock<std::mutex> lock(state_->batches.commit_mutex);
      immediate =
          state_->batches.next_commit_ticket == context->commit_ticket;
      state_->batches.commit_cv.wait(lock, [this, context] {
        return state_->batches.next_commit_ticket == context->commit_ticket;
      });
      context->commit_acquired = true;
    }
    std::lock_guard evidence_lock{state_->evidence.mutex};
    if (immediate) {
      ++::rund::detail::task::Stat(
          state_->evidence.metrics,
          ::rund::detail::task::StatSlot::SchedulerCommitWaitImmediate);
    } else {
      ++::rund::detail::task::Stat(
          state_->evidence.metrics,
          ::rund::detail::task::StatSlot::SchedulerCommitWaitBlocked);
    }
  } else if (context->commit_ticket == 0u) {
    context->commit_acquired = true;
  }
  if (!context->root_submit_recorded) {
    std::lock_guard lock{state_->evidence.mutex};
    ++::rund::detail::task::Stat(state_->evidence.metrics,
                                 ::rund::detail::task::StatSlot::Resumed);
    ++::rund::detail::task::Stat(
        state_->evidence.metrics,
        ::rund::detail::task::StatSlot::ContextSwitches);
    context->pending_root_submit = true;
    ++::rund::detail::task::Stat(
        state_->evidence.metrics,
        ::rund::detail::task::StatSlot::SchedulerResumeRootSubmitDeferred);
    context->root_submit_recorded = true;
  }
}

void Scheduler::FlushDeferredHotPathEnsureSkips(
    SchedulerThreadContext &context) noexcept {
  if (context.deferred_hot_path_ensure_skips == 0u) {
    return;
  }
  {
    std::lock_guard lock{state_->evidence.mutex};
    ::rund::detail::task::Stat(
        state_->evidence.metrics,
        ::rund::detail::task::StatSlot::SchedulerHotPathEnsureSkips) +=
        context.deferred_hot_path_ensure_skips;
  }
  context.deferred_hot_path_ensure_skips = 0u;
}

void Scheduler::ReleaseQuantumCommit() noexcept {
  SchedulerThreadContext *const context = active_scheduler_context;
  if (context == nullptr || context->scheduler != this) {
    return;
  }
  if (!context->commit_acquired || context->commit_ticket == 0u) {
    return;
  }
  FlushPendingRootSubmit();
  if (LaneCommitContains(context->commit_ticket)) {
    const std::uint64_t ticket = context->commit_ticket;
    context->commit_acquired = false;
    CompleteLaneCommit(ticket);
    return;
  }
  {
    std::lock_guard<std::mutex> lock(state_->batches.commit_mutex);
    if (state_->batches.next_commit_ticket == context->commit_ticket) {
      ++state_->batches.next_commit_ticket;
    }
  }
  context->commit_acquired = false;
  state_->batches.commit_cv.notify_all();
}

void Scheduler::CompletePrimitiveCommit() noexcept {
  SchedulerThreadContext *const context = active_scheduler_context;
  if (context == nullptr || context->scheduler != this ||
      !context->commit_acquired || !context->split_primitive_packets) {
    return;
  }
  {
    std::lock_guard lock{state_->evidence.mutex};
    ++::rund::detail::task::Stat(
        state_->evidence.metrics,
        ::rund::detail::task::StatSlot::PrimitivePackets);
  }
}

} // namespace rund::node

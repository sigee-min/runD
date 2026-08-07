#include <rund/counter.hpp>
#include <rund/task/stats/slots.hpp>

#include "../../../state/segment.hpp"
#include "../../../state/model/task.hpp"
#include "../../../state/storage.hpp"

namespace rund::node {

void Scheduler::FlushLaneOwnedTerminalRange(
    LaneOwnedTerminalRange &terminal_range) noexcept {
  if (!terminal_range.active) {
    return;
  }
  RecordTerminalRangeBatch(
      terminal_range.kind, terminal_range.code, terminal_range.first_task_id,
      terminal_range.last_task_id, terminal_range.first_ticket,
      terminal_range.last_ticket, terminal_range.logical_tasks,
      terminal_range.order_hash, true);
  terminal_range = LaneOwnedTerminalRange{};
}

void Scheduler::ExtendLaneOwnedTerminalRange(
    LaneOwnedTerminalRange &terminal_range,
    const ::rund::detail::task::OperationKind kind, const ReasonCode code,
    const std::uint64_t task_id, const std::uint64_t ticket) noexcept {
  const bool can_extend = terminal_range.active &&
                          terminal_range.kind == kind &&
                          terminal_range.code == code &&
                          terminal_range.last_task_id + 1u == task_id;
  if (!can_extend) {
    FlushLaneOwnedTerminalRange(terminal_range);
    terminal_range.active = true;
    terminal_range.kind = kind;
    terminal_range.code = code;
    terminal_range.first_task_id = task_id;
    terminal_range.first_ticket = ticket;
    terminal_range.order_hash = kFnvOffset;
  }
  terminal_range.last_task_id = task_id;
  terminal_range.last_ticket = ticket;
  ++terminal_range.logical_tasks;
  MixHash(terminal_range.order_hash, state_->plan.task(task_id));
  MixHash(terminal_range.order_hash, state_->plan.ticket(ticket));
  MixHash(terminal_range.order_hash, static_cast<std::uint64_t>(kind));
  MixHash(terminal_range.order_hash, static_cast<std::uint64_t>(code));
  MixHash(terminal_range.order_hash, terminal_range.logical_tasks);
}

bool Scheduler::CommitLaneOwnedTerminalEffect(
    LaneOwnedTerminalRange &terminal_range,
    const LaneSegmentEffect &effect) noexcept {
  if (effect.terminal &&
      effect.terminal_kind == ::rund::detail::task::OperationKind::Complete) {
    effect.record->state = TaskState::Completed;
    effect.record->failure_code = ReasonCode::Ok;
    effect.record->quantum_active = false;
    DestroyLaneCallable(*effect.record);
    ::rund::detail::counter::Accumulate(
        ::rund::detail::task::Stat(state_->evidence.metrics,
                                   ::rund::detail::task::StatSlot::Completed),
        1u);
    ++::rund::detail::task::Stat(
        state_->evidence.metrics,
        ::rund::detail::task::StatSlot::LaneLocalTasksCompleted);
    ++::rund::detail::task::Stat(
        state_->evidence.metrics,
        ::rund::detail::task::StatSlot::LaneLocalRetireCandidates);
    ExtendLaneOwnedTerminalRange(
        terminal_range, ::rund::detail::task::OperationKind::Complete,
        ReasonCode::Ok, effect.task_id, effect.logical_ticket);
    WakeJoinWaiters(effect.task_id, ReasonCode::Ok);
    DestroyTask(*effect.record);
    return true;
  }
  if (effect.terminal &&
      effect.terminal_kind == ::rund::detail::task::OperationKind::Fail) {
    effect.record->state = TaskState::Failed;
    effect.record->failure_code = effect.code;
    effect.record->quantum_active = false;
    DestroyLaneCallable(*effect.record);
    ++::rund::detail::task::Stat(state_->evidence.metrics,
                                 ::rund::detail::task::StatSlot::Failed);
    ++::rund::detail::task::Stat(
        state_->evidence.metrics,
        ::rund::detail::task::StatSlot::LaneLocalTasksCompleted);
    ++::rund::detail::task::Stat(
        state_->evidence.metrics,
        ::rund::detail::task::StatSlot::LaneLocalRetireCandidates);
    ExtendLaneOwnedTerminalRange(
        terminal_range, ::rund::detail::task::OperationKind::Fail, effect.code,
        effect.task_id, effect.logical_ticket);
    WakeJoinWaiters(effect.task_id, effect.code);
    DestroyTask(*effect.record);
    return true;
  }
  return false;
}

} // namespace rund::node

#include "local.hpp"

#include <cstdlib>

namespace rund::node {

LaneSegmentEffect
LaneWorkerAccess::MakeInitialEffect(TaskRecord &record,
                                    const LaneSegmentJob &job) noexcept {
  return LaneSegmentEffect{
      .record = &record,
      .task_id = record.id,
      .logical_ticket = job.logical_ticket,
      .canonical_index = job.canonical_index,
      .terminal_kind = ::rund::detail::task::OperationKind::None,
      .code = ReasonCode::Ok,
      .terminal = false,
      .trapped = false,
      .trap_kind = ::rund::detail::task::OperationKind::PrimitiveTrap,
      .trap_code = ReasonCode::Ok,
  };
}

void LaneWorkerAccess::RecordCompletedTerminal(const LaneSegmentEffect &effect,
                                               LaneJobFrame &frame) noexcept {
  if (frame.segment_first_task_id == 0u) {
    frame.segment_first_task_id = effect.task_id;
    frame.segment_first_ticket = effect.logical_ticket;
  }
  frame.segment_last_task_id = effect.task_id;
  frame.segment_last_ticket = effect.logical_ticket;
}

bool LaneWorkerAccess::RunSegmentTask(Scheduler &scheduler, TaskLane &lane,
                                      LaneJobFrame &frame,
                                      const LaneSegmentJob &job) noexcept {
  TaskRecord *const candidate =
      job.record != nullptr && job.record->id == job.task_id
          ? job.record
          : scheduler.state_->Find(job.task_id);
  if (candidate == nullptr || candidate->coroutine_task ||
      candidate->state != TaskState::Ready) {
    std::abort();
  }
  TaskRecord *const segment_record = candidate;
  segment_record->quantum_active = true;
  segment_record->state = TaskState::Running;
  struct QuantumExit final {
    TaskRecord &record;
    ~QuantumExit() { record.quantum_active = false; }
  } quantum_exit{*segment_record};

  LaneSegmentEffect effect = MakeInitialEffect(*segment_record, job);
  scheduler.RunLeafQuantum(*segment_record, job.logical_ticket,
                           job.split_primitive_packets, true, false, &effect);

  if (effect.terminal_kind ==
      ::rund::detail::task::OperationKind::Fail) {
    frame.segment_all_completed = false;
    frame.segment_has_trap_or_failure = true;
  } else if (effect.trapped) {
    frame.segment_all_completed = false;
    frame.segment_has_trap_or_failure = true;
  }

  if (effect.terminal &&
      effect.terminal_kind == ::rund::detail::task::OperationKind::Complete) {
    RecordCompletedTerminal(effect, frame);
  }
  const bool summarized_success =
      frame.segment_result_view_enabled && effect.terminal &&
      effect.terminal_kind == ::rund::detail::task::OperationKind::Complete &&
      effect.code == ReasonCode::Ok && !effect.trapped;
  if (!summarized_success) {
    if (lane.segment_effects.size() == lane.segment_effects.capacity()) {
      std::abort();
    }
    lane.segment_effects.push_back(effect);
  }
  ++frame.segment_completed;
  return effect.trapped;
}

void LaneWorkerAccess::RunSegmentSideExit(Scheduler &scheduler, TaskLane &lane,
                                          LaneJobFrame &frame) noexcept {
  for (std::size_t cursor = 0u; cursor < frame.segment_jobs.size(); ++cursor) {
    const LaneSegmentJob job = frame.segment_jobs[cursor];
    if (RunSegmentTask(scheduler, lane, frame, job)) {
      break;
    }
  }
}

void LaneWorkerAccess::RunSegmentJob(Scheduler &scheduler, TaskLane &lane,
                                     LaneJobFrame &frame) noexcept {
  frame.segment_original_job_count = frame.segment_jobs.size();
  frame.segment_all_completed = !frame.segment_jobs.empty();
  RunSegmentSideExit(scheduler, lane, frame);
}

void LaneWorkerAccess::Run(Scheduler &scheduler, TaskLane &lane,
                           LaneJobFrame &frame) noexcept {
  if (frame.segment_job) {
    RunSegmentJob(scheduler, lane, frame);
    return;
  }
  if (frame.work != nullptr) {
    if (frame.work->invoke != nullptr) {
      frame.work->invoke(frame.work->context);
    }
    return;
  }
  if (frame.record != nullptr) {
    scheduler.RunTaskQuantum(*frame.record, frame.ticket,
                             frame.split_primitive_packets, false,
                             frame.root_exclusive_commit, &frame.completion);
  }
}

void RunLaneJob(Scheduler &scheduler, TaskLane &lane,
                LaneJobFrame &frame) noexcept {
  LaneWorkerAccess::Run(scheduler, lane, frame);
}

} // namespace rund::node

#pragma once

#include "../../state/model/lane.hpp"
#include "../../state/model/task.hpp"
#include "../../state/model/work.hpp"
#include "../../state/storage.hpp"

#include <cstddef>
#include <cstdint>
#include <vector>

namespace rund::node {

struct LaneJobFrame {
  std::uint64_t id = 0u;
  TaskRecord *record = nullptr;
  std::uint64_t ticket = 0u;
  std::uint64_t job_sequence = 0u;
  bool direct_job = false;
  bool mailbox_job = false;
  bool split_primitive_packets = false;
  bool root_exclusive_commit = false;
  bool segment_job = false;
  bool segment_result_view_enabled = false;
  bool root_exclusive_hot_standby = false;
  std::uint64_t ready_signal_at_job_start = 0u;
  SchedulerWork *work = nullptr;
  CompletionLease completion{};

  std::vector<LaneSegmentJob> segment_jobs{};
  std::size_t segment_original_job_count = 0u;
  std::size_t segment_completed = 0u;
  bool segment_all_completed = false;
  bool segment_has_trap_or_failure = false;
  std::uint64_t segment_first_task_id = 0u;
  std::uint64_t segment_last_task_id = 0u;
  std::uint64_t segment_first_ticket = 0u;
  std::uint64_t segment_last_ticket = 0u;
};

struct LaneWorkerNotifyFlags {
  bool completion = false;
  bool signal = false;
};

struct LaneWorkerAccess {
  [[nodiscard]] static bool Claim(TaskLane &lane, LaneJobFrame &frame) noexcept;
  static void Run(Scheduler &scheduler, TaskLane &lane,
                  LaneJobFrame &frame) noexcept;
  static void Publish(Scheduler &scheduler, TaskLane &lane,
                      LaneJobFrame &frame) noexcept;

private:
  static void ResetFrame(LaneJobFrame &frame) noexcept;
  static void LoadMailboxJob(TaskLane &lane, LaneJobFrame &frame) noexcept;
  static void LoadExternalWakeJob(TaskLane &lane, LaneJobFrame &frame) noexcept;
  static void LoadDirectReadyJob(TaskLane &lane, LaneJobFrame &frame) noexcept;
  static void LoadWork(TaskLane &lane, LaneJobFrame &frame) noexcept;
  static void LoadSegmentJob(TaskLane &lane, LaneJobFrame &frame) noexcept;
  static void LoadRegularJob(TaskLane &lane, LaneJobFrame &frame) noexcept;

  [[nodiscard]] static LaneSegmentEffect
  MakeInitialEffect(TaskRecord &record, const LaneSegmentJob &job) noexcept;
  static void RecordCompletedTerminal(const LaneSegmentEffect &effect,
                                      LaneJobFrame &frame) noexcept;
  [[nodiscard]] static bool RunSegmentTask(Scheduler &scheduler, TaskLane &lane,
                                           LaneJobFrame &frame,
                                           const LaneSegmentJob &job) noexcept;
  static void RunSegmentSideExit(Scheduler &scheduler, TaskLane &lane,
                                 LaneJobFrame &frame) noexcept;
  static void RunSegmentJob(Scheduler &scheduler, TaskLane &lane,
                            LaneJobFrame &frame) noexcept;

  static void NotifyLaneWaiters(TaskLane &lane,
                                LaneWorkerNotifyFlags flags) noexcept;
  static void PublishSegmentJob(TaskLane &lane, LaneJobFrame &frame) noexcept;
  static void ReleaseDirectJob(Scheduler &scheduler, TaskLane &lane,
                               bool mailbox) noexcept;
  static void PublishRegularJob(Scheduler &scheduler, TaskLane &lane,
                                const LaneJobFrame &frame) noexcept;
};

[[nodiscard]] bool ClaimLaneJob(TaskLane &lane, LaneJobFrame &frame) noexcept;
void RunLaneJob(Scheduler &scheduler, TaskLane &lane,
                LaneJobFrame &frame) noexcept;
void PublishLaneJob(Scheduler &scheduler, TaskLane &lane,
                    LaneJobFrame &frame) noexcept;

} // namespace rund::node

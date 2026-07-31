#pragma once

#include "segment.hpp"

#include <atomic>
#include <condition_variable>
#include <cstddef>
#include <cstdint>
#include <mutex>
#include <vector>

namespace rund::node {

struct SchedulerWork;
struct TaskRecord;

struct TaskLane {
  std::mutex mutex{};
  std::condition_variable ready{};
  std::condition_variable done_cv{};
  std::uint64_t task_id = 0u;
  TaskRecord *task_record = nullptr;
  std::uint64_t commit_ticket = 0u;
  std::uint64_t job_sequence = 0u;
  std::uint64_t next_job_sequence = 1u;
  std::atomic<std::uint64_t> completed_job_sequence{0u};
  std::atomic<std::uint64_t> ready_signal{0u};
  std::atomic<std::uint64_t> completed_job_signal{0u};
  bool completion_wait_requested = false;
  bool completion_signal_wait_requested = false;
  bool ready_wait_requested = false;
  std::vector<LaneSegmentJob> segment_jobs{};
  std::vector<LaneSegmentEffect> segment_effects{};
  std::uint64_t segment_first_task_id = 0u;
  std::uint64_t segment_last_task_id = 0u;
  std::uint64_t segment_first_ticket = 0u;
  std::uint64_t segment_last_ticket = 0u;
  bool segment_all_completed = false;
  bool segment_has_trap_or_failure = false;
  LaneSegmentResultView segment_result_view{};
  std::atomic<std::uint64_t> segment_result_sequence{0u};
  std::atomic<std::uint64_t> segment_completed_ticket{0u};
  bool segment_result_view_enabled = false;
  std::size_t segment_job_count = 0u;
  std::size_t segment_completed_jobs = 0u;
  bool split_primitive_packets = false;
  bool root_exclusive_commit = false;
  bool direct_job = false;
  bool root_exclusive_hot_standby = false;
  bool has_job = false;
  bool running = false;
  bool accepts_nested_job = false;
  bool nested_job_active = false;
  std::atomic<bool> root_reserved{false};
  bool stop = false;
  std::atomic<std::uint8_t> mailbox_state{0u};
  std::atomic<std::uint64_t> mailbox_task_id{0u};
  std::atomic<std::uint64_t> mailbox_commit_ticket{0u};
  std::atomic<std::uint64_t> mailbox_job_sequence{0u};
  std::atomic<TaskRecord *> mailbox_record{nullptr};
  TaskRecord *external_wake_head = nullptr;
  TaskRecord *external_wake_tail = nullptr;
  TaskRecord *direct_ready_head = nullptr;
  TaskRecord *direct_ready_tail = nullptr;
  SchedulerWork *work_head = nullptr;
  SchedulerWork *work_tail = nullptr;
};

extern thread_local TaskLane *active_task_lane;

[[nodiscard]] bool TakeLaneCompletion(TaskLane &target,
                                      std::uint64_t sequence) noexcept;
void PublishLaneCompletion(TaskLane &target, std::uint64_t sequence) noexcept;

} // namespace rund::node

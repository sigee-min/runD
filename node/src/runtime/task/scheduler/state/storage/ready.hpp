#include "ready/queue.hpp"

struct SchedulerReadyState {
  SchedulerReadyState() noexcept;
  ~SchedulerReadyState();

  ReadyQueue ready{};
  std::size_t ready_depth = 0u;
  std::vector<TaskRecord> records{};
  std::vector<std::uint32_t> record_index_slots{};
  std::size_t record_index_size = 0u;
  std::size_t record_index_deleted = 0u;
  std::uint32_t free_record_head = 0u;
  std::vector<TimerWait> timers{};
  std::vector<TimerWaitIdIndexEntry> timer_wait_id_index{};
  std::vector<JoinWait> join_waits{};
};

struct SchedulerBatchState {
  std::uint64_t root_single_join_streak = 0u;
  std::size_t root_single_join_session_lane = 0u;
  bool root_single_join_session_lane_active = false;

  struct TaskSpawnBatch {
    bool active = false;
    std::uint64_t first_task_id = 0u;
    std::uint64_t last_task_id = 0u;
    std::uint64_t parent_task_id = 0u;
    std::uint64_t scope_id = 0u;
    std::uint64_t logical_count = 0u;
    std::uint64_t order_hash = kFnvOffset;
  } task_spawn_batch{};

  struct RootSingleJoinEpoch {
    bool active = false;
    std::uint64_t epoch_id = 0u;
    std::uint64_t first_task_id = 0u;
    std::uint64_t last_task_id = 0u;
    std::uint64_t parent_task_id = 0u;
    std::uint64_t scope_id = 0u;
    std::uint64_t logical_tasks = 0u;
    std::uint64_t logical_events = 0u;
    std::uint64_t spawn_participant_hash = kFnvOffset;
  } root_single_join_epoch{};

  struct YieldResumeBatch {
    bool active = false;
    std::uint64_t task_id = 0u;
    std::uint64_t logical_yields = 0u;
    std::uint64_t logical_count = 0u;
  } yield_resume_batch{};

  struct SpawnIdReservation {
    bool active = false;
    std::uint64_t parent_task_id = 0u;
    std::uint64_t scope_id = 0u;
    std::uint64_t next_task_id = 0u;
    std::uint64_t remaining = 0u;
    bool index_materialized = false;
  } spawn_id_reservation{};

  struct PendingRootJoinRange {
    bool active = false;
    std::uint64_t first_task_id = 0u;
    std::uint64_t last_task_id = 0u;
    std::uint64_t logical_tasks = 0u;
    ReasonCode code = ReasonCode::Ok;
  } pending_root_join_range{};

  struct LaneResidualJoinOwnerPolicy {
    bool active = false;
    std::uint64_t first_task_id = 0u;
    std::uint64_t last_task_id = 0u;
  } lane_residual_join_owner_policy{};

  std::mutex commit_mutex{};
  std::condition_variable commit_cv{};
  std::uint64_t next_commit_ticket = 1u;
  std::uint64_t next_commit_ticket_to_issue = 1u;
  std::atomic<std::uint64_t> lane_commit_first{0u};
  std::atomic<std::uint64_t> lane_commit_frontier{0u};
  std::atomic<std::uint64_t> lane_commit_end{0u};
  std::atomic<std::uint32_t> lane_commit_waiters{0u};
  std::atomic<bool> lane_commit_active{false};
  std::mutex direct_mutex{};
  std::condition_variable direct_cv{};
  std::uint64_t ready_epoch = 0u;
  std::uint64_t consumed_ready_epoch = 0u;
  std::atomic<std::uint32_t> direct_jobs_in_flight{0u};
  std::atomic<std::uint32_t> task_direct_jobs_in_flight{0u};
};

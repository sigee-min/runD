struct SchedulerLaneState {
  SchedulerLaneState() noexcept;
  ~SchedulerLaneState();

  std::vector<std::unique_ptr<TaskLane>> lanes{};
  std::vector<std::thread> lane_threads{};
  std::vector<std::uint8_t> lane_participated{};
  std::vector<std::uint8_t> lane_batch_used_scratch{};
  std::vector<std::uint64_t> ready_batch_scratch{};
  std::vector<std::uint64_t> ready_deferred{};
  std::vector<LaneBatchSubmission> submitted_batch_scratch{};
  std::vector<std::uint32_t> segment_commit_lanes{};
  ::rund::kernel::ParallelRuntimeProvider kernel_provider{};
  bool lanes_started = false;
  ReasonCode lane_code = ReasonCode::TaskWorkersInvalid;
};

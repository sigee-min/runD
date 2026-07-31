struct SchedulerResourceState {
  ::rund::SchedulerConfig limits{};
  std::uint32_t live_channels = 0u;
  std::uint32_t live_channel_buffer_slots = 0u;
  std::uint32_t live_channel_waits = 0u;
  FrameArena frame_arena{};
  CallablePool callable_pool{};
  CompletionPool completion_pool{};
  std::vector<std::size_t> join_slots{};
  std::atomic<std::uint64_t> live_tasks{0u};
  ::rund::PreparedMemory prepared_memory{};
};

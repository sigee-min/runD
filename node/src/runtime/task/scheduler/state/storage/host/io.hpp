struct HostIoSlot;

struct SchedulerHostIoState {
  SchedulerHostIoState() noexcept;
  ~SchedulerHostIoState();

  std::mutex mutex{};
  std::condition_variable ready{};
  std::unique_ptr<HostIoSlot[]> slots{};
  std::size_t capacity = 0u;
  HostIoSlot *free_head = nullptr;
  HostIoSlot *queue_head = nullptr;
  HostIoSlot *queue_tail = nullptr;
  std::thread worker{};
  std::size_t submissions_in_progress = 0u;
  std::uint64_t next_submission_sequence = 1u;
  std::uint64_t next_execution_sequence = 1u;
  bool stop = false;
};

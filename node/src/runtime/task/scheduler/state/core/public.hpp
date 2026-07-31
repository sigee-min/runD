  Scheduler();
  ~Scheduler();
  Scheduler(const Scheduler &) = delete;
  Scheduler &operator=(const Scheduler &) = delete;
  Scheduler(Scheduler &&) = delete;
  Scheduler &operator=(Scheduler &&) = delete;

  [[nodiscard]] static task::Handle InvalidHandle(ReasonCode code) noexcept;

  [[nodiscard]] bool Configure(
      ::rund::SchedulerConfig config,
      rund::kernel::ParallelRuntimeProvider provider,
      ::rund::ReplayConfig replay,
      ::rund::host::random::RunSeed random_seed);

  [[nodiscard]] static Scheduler *Active() noexcept { return active_; }
  static void SetActive(Scheduler *scheduler) noexcept;
  [[nodiscard]] bool EnqueueWork(SchedulerWork* work,
                                 std::uint32_t worker) noexcept;

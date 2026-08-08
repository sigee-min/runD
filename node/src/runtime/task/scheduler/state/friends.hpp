friend struct task::Handle;
friend class ::rund::host::io::ReadOp;
friend class ::rund::host::io::WriteOp;
friend class ::rund::detail::task::AwaitAccess;
friend class ::rund::detail::task::ChannelAccess;
friend class ::rund::detail::task::ApiAccess;
friend ::rund::SchedulerConfig scheduler_access::ActiveLimits() noexcept;
friend std::uint32_t
scheduler_access::CoroutineFrameByteLimit(const Scheduler &scheduler) noexcept;
friend bool ReactorCleanupWait(Scheduler &scheduler,
                               ReactorCleanupRequest request) noexcept;
friend bool
ReactorCleanupRemovedWait(Scheduler &scheduler,
                          ReactorRemovedWaitCleanupRequest request) noexcept;
friend void reactor_cancel_cleanup::WakeTask(
    Scheduler &scheduler, TaskRecord &record, std::uint64_t wait_id,
    ReactorHandle handle, ReactorInterest interest, ReactorEvent events,
    std::int64_t deadline_ns, ReasonCode reason, bool cleanup_ok) noexcept;
friend bool reactor_cancel_cleanup::CancelTimeoutTimer(
    Scheduler &scheduler, std::uint64_t wait_id,
    ReactorTimeoutCleanupPolicy policy) noexcept;
friend bool reactor_cancel_cleanup::CleanupGroup(
    Scheduler &scheduler, const ReactorCleanupRequest &request) noexcept;
friend bool reactor_cancel_cleanup::CleanupSingleWait(
    Scheduler &scheduler, ReactorCleanupRequest request) noexcept;
friend struct LaneWorkerAccess;
friend struct ReadyManyAccess;
friend ReasonCode ReactorCloseInvalidateFd(Scheduler &scheduler,
                                           int fd) noexcept;
friend ReasonCode
ReactorGenerationCleanupStaleWaits(Scheduler &scheduler, ReactorHandle fd,
                                   std::uint64_t current_generation) noexcept;
friend bool ReactorGenerationCleanupInvalidWaits(Scheduler &scheduler,
                                                 bool *invalidated) noexcept;

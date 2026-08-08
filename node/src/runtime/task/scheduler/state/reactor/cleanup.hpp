struct ReactorInvariantSnapshot;
struct ReactorCleanupRequest;
struct ReactorRemovedWaitCleanupRequest;
enum class ReactorTimeoutCleanupPolicy : std::uint8_t;
[[nodiscard]] bool ReactorCleanupWait(Scheduler& scheduler,
                                      ReactorCleanupRequest request) noexcept;
[[nodiscard]] bool ReactorCleanupRemovedWait(
    Scheduler& scheduler, ReactorRemovedWaitCleanupRequest request) noexcept;

namespace reactor_cancel_cleanup {
void WakeTask(Scheduler& scheduler, TaskRecord& record, std::uint64_t wait_id,
               ReactorHandle handle, ReactorInterest interest,
               ReactorEvent events,
               std::int64_t deadline_ns,
               ReasonCode reason, bool cleanup_ok) noexcept;
[[nodiscard]] bool
CancelTimeoutTimer(Scheduler &scheduler, std::uint64_t wait_id,
                   ReactorTimeoutCleanupPolicy policy) noexcept;
[[nodiscard]] bool CleanupGroup(Scheduler& scheduler,
                                const ReactorCleanupRequest& request) noexcept;
[[nodiscard]] bool CleanupSingleWait(Scheduler& scheduler,
                                     ReactorCleanupRequest request) noexcept;
}  // namespace reactor_cancel_cleanup

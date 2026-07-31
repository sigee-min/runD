class Scheduler;
struct ScopeEvidence;
struct ReadyManyEntry;
struct ReadyManyAccess;
struct LaneJobFrame;
struct LaneWorkerAccess;
struct ExternalWake;
struct LaneOwnedSegmentSummary;
struct LaneOwnedTerminalRange;
struct LaneSegmentJob;
struct SchedulerThreadContext;
struct SchedulerWork;
struct TaskLane;
enum class HostIoKind : std::uint8_t;
struct HostIoOperation;
struct HostIoCompletion;
namespace scheduler_access {
[[nodiscard]] ::rund::SchedulerConfig ActiveLimits() noexcept;
[[nodiscard]] std::uint32_t
CoroutineFrameByteLimit(const Scheduler &scheduler) noexcept;
} // namespace scheduler_access

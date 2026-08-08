[[nodiscard]] bool RunOnLane(std::uint64_t id, std::uint64_t commit_ticket = 0u,
                             bool split_primitive_packets = false,
                             bool root_exclusive_commit = false,
                             bool root_exclusive_hot_standby = false) noexcept;
[[nodiscard]] bool
RunReadyBatch(const std::vector<std::uint64_t> &task_ids) noexcept;
[[nodiscard]] bool
RunLaneOwnedSegments(const std::vector<std::uint64_t> &task_ids) noexcept;
void RestoreLaneOwnedSegmentTasks(
    const std::vector<std::uint64_t> &task_ids) noexcept;
[[nodiscard]] bool
PlanLaneOwnedSegments(const std::vector<std::uint64_t> &task_ids,
                      std::vector<LaneOwnedSegmentLane> &segments,
                      std::size_t *planned_logical_tasks) noexcept;
[[nodiscard]] std::size_t
PublishLaneOwnedSegments(std::vector<LaneOwnedSegmentLane> &segments,
                         std::size_t planned_logical_tasks) noexcept;
void RecordLaneOwnedSegmentAdmission(std::size_t submitted_lanes,
                                     std::size_t logical_tasks) noexcept;
void NotifyLaneOwnedSegments(const std::vector<LaneOwnedSegmentLane> &segments,
                             std::size_t submitted_lanes) noexcept;
void DrainLaneOwnedSegmentResults(std::vector<LaneOwnedSegmentLane> &segments,
                                  LaneOwnedSegmentSummary *summary) noexcept;
[[nodiscard]] bool CommitSuccessfulLaneOwnedSegments(
    const std::vector<std::uint64_t> &task_ids, std::size_t logical_tasks,
    const LaneOwnedSegmentSummary &summary) noexcept;
void CollectLaneOwnedSegmentEffects(
    const std::vector<LaneOwnedSegmentLane> &segments,
    std::vector<std::uint64_t> &executed_task_ids,
    std::vector<LaneSegmentEffect> &lane_effects) noexcept;
void FlushLaneOwnedTerminalRange(
    LaneOwnedTerminalRange &terminal_range) noexcept;
void ExtendLaneOwnedTerminalRange(LaneOwnedTerminalRange &terminal_range,
                                  ::rund::detail::task::OperationKind kind,
                                  ReasonCode code, std::uint64_t task_id,
                                  std::uint64_t ticket) noexcept;
void CommitLaneOwnedSegmentEffects(
    const std::vector<LaneSegmentEffect> &lane_effects) noexcept;
[[nodiscard]] bool
CommitLaneOwnedTerminalEffect(LaneOwnedTerminalRange &terminal_range,
                              const LaneSegmentEffect &effect) noexcept;
void CommitLaneOwnedPrimitiveTrapEffect(
    LaneOwnedTerminalRange &terminal_range,
    const LaneSegmentEffect &effect) noexcept;
void MarkLaneOwnedSegmentSideExits(
    const std::vector<std::uint64_t> &task_ids,
    const std::vector<std::uint64_t> &executed_task_ids) noexcept;
void RequeueLaneOwnedSegmentSideExits(
    const std::vector<std::uint64_t> &task_ids) noexcept;
[[nodiscard]] bool CanSubmitToLane(std::uint64_t id) noexcept;
[[nodiscard]] bool DispatchQueuedReady(std::uint64_t id) noexcept;
[[nodiscard]] std::uint32_t LaneCompletionSpinLoadBudget() const noexcept;
[[nodiscard]] std::uint32_t LaneCompletionSpinYieldStride() const noexcept;
[[nodiscard]] std::uint32_t LaneHotStandbySpinBudget() const noexcept;
[[nodiscard]] std::chrono::microseconds NestedJoinPollInterval() const noexcept;
[[nodiscard]] bool
TryConsumeCompletedJobSpin(TaskLane &lane, std::uint64_t sequence,
                           bool release_root_reservation = false) noexcept;
[[nodiscard]] bool
TryConsumeLaneCompletionGroupSignal(TaskLane &lane,
                                    std::uint64_t sequence) noexcept;
[[nodiscard]] bool WaitForLaneCompletionSignal(TaskLane &lane,
                                               std::uint64_t sequence) noexcept;
[[nodiscard]] bool StartLanes() noexcept;
void StopLanes() noexcept;
[[nodiscard]] std::uint64_t PopReady(std::uint64_t only_scope_id) noexcept;
void WakeDueTimers() noexcept;
[[nodiscard]] TimerDeadline
MakeTimerDeadline(std::chrono::nanoseconds duration) const noexcept;
[[nodiscard]] bool CancelReactorTimeoutTimer(std::uint64_t wait_id) noexcept;
[[nodiscard]] bool WakeReactorTimeout(const TimerWait &wait) noexcept;
[[nodiscard]] bool WakeReactorManyTimeout(const TimerWait &wait) noexcept;
[[nodiscard]] ::rund::net::ready::many::Wait WaitReactorManyPrepared(
    std::span<const ReactorManyRequest> requests,
    std::span<::rund::net::ready::Event> out,
    std::optional<std::chrono::nanoseconds> timeout,
    ::rund::net::ready::many::Budget budget,
    ::rund::detail::task::StopIdentity stop = {},
    ::rund::net::ready::Set ready_set = {}) noexcept;
[[nodiscard]] bool
CopyReactorManyEvents(std::uint64_t group_id,
                      std::span<::rund::net::ready::Event> out,
                      std::uint32_t *copied) noexcept;
[[nodiscard]] bool WakeReactorManyGroupFromWait(const ReactorWait &wait,
                                                ReasonCode code,
                                                ReactorEvent events,
                                                bool store_event) noexcept;
[[nodiscard]] bool CancelReadySetWaitGroups(::rund::net::ready::Set set,
                                            ReasonCode code) noexcept;
[[nodiscard]] int TimerBoundIoPollTimeoutMs() const noexcept;
[[nodiscard]] bool DrainReadyReactor(int timeout_ms,
                                     bool force_apply = false) noexcept;
[[nodiscard]] bool
DrainReactorReadyBatch(
    const std::vector<ReactorReady> &ordered,
    ReactorInvalidChangeToken invalid_change) noexcept;
void WakeReadyReactor() noexcept;
[[nodiscard]] bool WaitForDirectJobs() noexcept;
[[nodiscard]] bool
WakeDeadlockedTasks(std::uint64_t only_scope_id = 0u) noexcept;

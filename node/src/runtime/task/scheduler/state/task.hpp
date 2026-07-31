class ControlCommitScope;

void RunTaskQuantum(TaskRecord &record, std::uint64_t commit_ticket,
                    bool split_primitive_packets, bool lane_owned_segment,
                    bool root_exclusive_commit,
                    CompletionLease *completion = nullptr) noexcept;
void RunCoroutineQuantum(TaskRecord &record, std::uint64_t commit_ticket,
                         bool split_primitive_packets, bool lane_owned_segment,
                         bool root_exclusive_commit,
                         CompletionLease *completion) noexcept;
[[nodiscard]] bool ResumeCoroutine(TaskRecord &record) noexcept;
void RunLeafQuantum(TaskRecord &record, std::uint64_t commit_ticket,
                    bool split_primitive_packets, bool lane_owned_segment,
                    bool root_exclusive_commit,
                    LaneSegmentEffect *lane_effect = nullptr) noexcept;
[[nodiscard]] bool TryResumeSameLane(TaskRecord &record,
                                     SchedulerThreadContext &context) noexcept;
[[nodiscard]] bool EnqueueExternalWake(ExternalWake wake) noexcept;
void QueueDirect(TaskLane &lane, TaskRecord &record, std::uint64_t ticket,
                 std::uint64_t sequence) noexcept;
[[nodiscard]] TaskRecord *PrepareLaneQuantum(std::uint64_t id) noexcept;
[[nodiscard]] TaskRecord *PrepareLaneQuantum(TaskRecord &record) noexcept;
void FinishQuantum(TaskRecord &record) noexcept;
[[nodiscard]] bool Matches(const task::Handle &handle,
                           const TaskRecord *record) const noexcept;
[[nodiscard]] bool TerminalAt(const task::Handle *handles,
                              const std::size_t *slots,
                              std::size_t count) const noexcept;
[[nodiscard]] ReasonCode FailureCodeAt(const task::Handle *handles,
                                       const std::size_t *slots,
                                       std::size_t count) const noexcept;
void DestroyTask(TaskRecord &record) noexcept;
void DestroyLaneCallable(TaskRecord &record) noexcept;
void RetireTaskRecord(TaskRecord &record, bool count_record_retire) noexcept;
void RetireTask(TaskRecord &record) noexcept;
void QueueRootJoinRetireKnown(TaskRecord &record, std::uint64_t task_id,
                              ReasonCode code) noexcept;
void FlushPendingRootJoinRetireBatch() noexcept;
[[nodiscard]] ReasonCode ValidateSpawnBudget(ReadyAdmission admission) noexcept;
[[nodiscard]] std::uint64_t CurrentTaskId() const noexcept;
[[nodiscard]] std::uint64_t CurrentScopeId() const noexcept;
void SetCurrentScopeId(std::uint64_t scope_id) noexcept;
[[nodiscard]] std::uint64_t IssueCommitTicket() noexcept;
[[nodiscard]] std::uint64_t
IssueCommitTickets(std::uint64_t logical_events) noexcept;
[[nodiscard]] std::uint64_t
IssueLaneCommitTickets(std::uint64_t logical_events) noexcept;
[[nodiscard]] bool LaneCommitContains(std::uint64_t ticket) const noexcept;
[[nodiscard]] bool WaitLaneCommit(std::uint64_t ticket) noexcept;
void CompleteLaneCommit(std::uint64_t ticket) noexcept;
void AdvanceLaneCommit() noexcept;
void EnsureCurrentCommit() noexcept;
void FlushDeferredHotPathEnsureSkips(SchedulerThreadContext &context) noexcept;
void CompletePrimitiveCommit() noexcept;
[[nodiscard]] ReasonCode RejectPrimitive() noexcept;
void SetLeafFailure(TaskRecord &record, ReasonCode code) noexcept;
[[nodiscard]] bool TrapLaneOwnedSegmentPrimitive(
    ::rund::detail::task::OperationKind kind =
        ::rund::detail::task::OperationKind::PrimitiveTrap,
    ReasonCode code = ReasonCode::Ok) noexcept;
void ReleaseQuantumCommit() noexcept;

[[nodiscard]] bool Step(std::uint64_t only_scope_id = 0u) noexcept;
[[nodiscard]] std::size_t ReadyDepth() const noexcept;
[[nodiscard]] bool ReadyQueuesEmpty() const noexcept;
void RestoreReadyFront(std::uint64_t id,
                       std::uint64_t only_scope_id = 0u) noexcept;
[[nodiscard]] bool RequeueReadyTask(
    std::uint64_t id, std::uint64_t only_scope_id = 0u) noexcept;
[[nodiscard]] ReadyPick
PopSubmittableReady(std::uint64_t only_scope_id) noexcept;
[[nodiscard]] ReadyPick
WaitUntilProgressReady(std::uint64_t only_scope_id) noexcept;
[[nodiscard]] bool DispatchReadyTask(std::uint64_t id,
                                     std::uint64_t only_scope_id) noexcept;
[[nodiscard]] bool RunMultiLaneReadyBatch(std::uint64_t id) noexcept;
[[nodiscard]] bool RunRootSingleJoinReadyTarget(std::uint64_t id) noexcept;
void ValidateReplayDrain() noexcept;
[[nodiscard]] task::Status JoinManyWithSlots(const task::Handle *handles,
                                           std::size_t count) noexcept;
[[nodiscard]] task::Status
JoinManyWithProvidedSlots(const task::Handle *handles, std::size_t count,
                          std::size_t *join_slots) noexcept;
void WakeJoinWaiters(std::uint64_t target_task_id, ReasonCode result) noexcept;

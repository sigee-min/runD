void Record(::rund::detail::task::OperationKind kind, ReasonCode code,
            std::uint64_t task_id = 0u, std::uint64_t target_id = 0u,
            std::uint64_t wait_id = 0u, std::uint64_t channel_id = 0u,
            int fd = -1, short interest = 0, short revents = 0,
            std::int64_t deadline_ns = 0, std::uint64_t value_count = 0u,
            std::uint64_t match_sequence = 0u, std::uint64_t baton_epoch = 0u,
            std::uint64_t task_op_ordinal = 0u,
            std::uint64_t target_op_ordinal = 0u, std::uint64_t region_id = 0u,
            std::uint64_t epoch_id = 0u, std::uint64_t logical_count = 0u,
            std::uint64_t order_hash = 0u,
            ReasonCode side_exit_code = ReasonCode::Ok) noexcept;
void RecordPhysical(
    ::rund::detail::task::OperationKind kind, ReasonCode code,
    std::uint64_t task_id = 0u, std::uint64_t target_id = 0u,
    std::uint64_t wait_id = 0u, std::uint64_t channel_id = 0u, int fd = -1,
    short interest = 0, short revents = 0, std::int64_t deadline_ns = 0,
    std::uint64_t value_count = 0u, std::uint64_t match_sequence = 0u,
    std::uint64_t baton_epoch = 0u, std::uint64_t task_op_ordinal = 0u,
    std::uint64_t target_op_ordinal = 0u, std::uint64_t region_id = 0u,
    std::uint64_t epoch_id = 0u, std::uint64_t logical_count = 0u,
    std::uint64_t order_hash = 0u,
    ReasonCode side_exit_code = ReasonCode::Ok) noexcept;
void FlushPendingRootSubmit() noexcept;
[[nodiscard]] bool ConsumePendingRootSubmit(std::uint64_t task_id) noexcept;
[[nodiscard]] bool
RecordRootSingleJoinEpochTask(::rund::detail::task::OperationKind terminal_kind,
                              ReasonCode code, std::uint64_t task_id) noexcept;
void FlushRootSingleJoinEpoch(
    ReasonCode side_exit_code = ReasonCode::Ok) noexcept;
void RecordSpawnBatch(std::uint64_t task_id, std::uint64_t parent_task_id,
                      std::uint64_t scope_id, std::uint64_t name_hash) noexcept;
void FlushTaskSpawnBatch(ReasonCode side_exit_code = ReasonCode::Ok) noexcept;
void FlushYieldBatch(ReasonCode side_exit_code = ReasonCode::Ok) noexcept;
void RecordLaneLocalYieldBatch(std::uint64_t task_id,
                               std::uint64_t logical_yields,
                               std::uint64_t logical_ticket) noexcept;
void RecordLaneLocalYieldEpochBatch(std::uint64_t first_task_id,
                                    std::uint64_t last_task_id,
                                    std::uint64_t first_ticket,
                                    std::uint64_t last_ticket,
                                    std::uint64_t logical_tasks,
                                    std::uint64_t logical_yields,
                                    std::uint64_t participant_hash) noexcept;
void RecordTerminalBatch(::rund::detail::task::OperationKind terminal_kind,
                         ReasonCode code, std::uint64_t task_id) noexcept;
void RecordTerminalRangeBatch(::rund::detail::task::OperationKind terminal_kind,
                              ReasonCode code, std::uint64_t first_task_id,
                              std::uint64_t last_task_id,
                              std::uint64_t first_ticket,
                              std::uint64_t last_ticket,
                              std::uint64_t logical_tasks,
                              std::uint64_t order_hash,
                              bool includes_root_submit) noexcept;
void RecordYieldBatch(std::uint64_t task_id) noexcept;
void RecordJoinBatch(std::uint64_t target_task_id, ReasonCode code) noexcept;
void RecordJoinRetireBatch(std::uint64_t first_task_id,
                           std::uint64_t last_task_id, ReasonCode code,
                           std::uint64_t logical_tasks) noexcept;

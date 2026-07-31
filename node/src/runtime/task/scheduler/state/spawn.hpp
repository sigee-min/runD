enum class ReadyAdmission : std::uint8_t {
  Spawn,
  Continuation,
};

[[nodiscard]] task::Handle
SpawnPrepared(const char *name, ::rund::detail::task::Callable *callable,
              ::rund::detail::task::CoroutineStart coroutine,
              ::rund::detail::task::ResultRef *observer,
              ReadyAdmission admission) noexcept;
[[nodiscard]] ::rund::detail::task::Spawned
SpawnWithCompletion(const char *name,
                    ::rund::detail::task::CoroutineStart coroutine,
                    ReadyAdmission admission) noexcept;
[[nodiscard]] task::Handle
SpawnPreparedCommitted(const char *name,
                       ::rund::detail::task::Callable *callable,
                       ::rund::detail::task::CoroutineStart coroutine,
                       ::rund::detail::task::ResultRef *observer,
                       ReadyAdmission admission) noexcept;
void DestroyRejectedSpawnPayload(
    ::rund::detail::task::CoroutineStart coroutine) noexcept;
[[nodiscard]] std::uint64_t IssueSpawnTaskId(std::uint64_t parent_task_id,
                                             std::uint64_t scope_id) noexcept;
[[nodiscard]] std::size_t ClaimSpawnTaskSlot(bool &reuse_record) noexcept;
void ReleasePreparedSpawnTaskSlot(TaskRecord &record, std::size_t record_index,
                                  bool reuse_record) noexcept;
[[nodiscard]] bool MaterializePreparedSpawnTask(
    TaskRecord &record, ::rund::detail::task::Callable *callable,
    ::rund::detail::task::CoroutineStart coroutine, CompletionLease completion,
    std::uint64_t parent_task_id, std::uint64_t scope_id,
    bool root_spawn) noexcept;
[[nodiscard]] task::Handle
EnqueuePreparedSpawnTask(TaskRecord &record, std::size_t record_index,
                         bool reuse_record, std::uint64_t parent_id,
                         std::uint64_t name_hash,
                         ReadyAdmission admission) noexcept;

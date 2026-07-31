#pragma once

#include "../../../state/storage.hpp"

namespace rund::node {

[[nodiscard]] std::uint64_t Avalanche(std::uint64_t value) noexcept;
[[nodiscard]] std::uint64_t
CanonicalOrderHash(::rund::detail::task::OperationKind kind,
                   std::uint64_t a = 0u, std::uint64_t b = 0u,
                   std::uint64_t c = 0u, std::uint64_t d = 0u,
                   std::uint64_t e = 0u) noexcept;
[[nodiscard]] std::uint64_t
SpawnBatchOrderHash(std::uint64_t first_task_id, std::uint64_t last_task_id,
                    std::uint64_t parent_task_id, std::uint64_t scope_id,
                    std::uint64_t logical_count,
                    std::uint64_t participant_hash) noexcept;
[[nodiscard]] std::uint64_t
AppendSpawnParticipantHash(std::uint64_t participant_hash,
                           std::uint64_t task_id, std::uint64_t parent_task_id,
                           std::uint64_t scope_id,
                           std::uint64_t name_hash) noexcept;
[[nodiscard]] std::uint64_t
TerminalBatchOrderHash(bool includes_root_submit,
                       ::rund::detail::task::OperationKind terminal_kind,
                       ReasonCode code, std::uint64_t task_id,
                       std::uint64_t logical_count) noexcept;
[[nodiscard]] std::uint64_t TerminalRangeBatchOrderHash(
    bool includes_root_submit,
    ::rund::detail::task::OperationKind terminal_kind, ReasonCode code,
    std::uint64_t first_task_id, std::uint64_t last_task_id,
    std::uint64_t first_ticket, std::uint64_t last_ticket,
    std::uint64_t logical_tasks, std::uint64_t participant_hash) noexcept;
[[nodiscard]] std::uint64_t
YieldBatchOrderHash(bool includes_root_submit, std::uint64_t task_id,
                    std::uint64_t logical_count,
                    std::uint64_t logical_yields = 1u) noexcept;
[[nodiscard]] std::uint64_t LaneLocalYieldEpochOrderHash(
    std::uint64_t first_task_id, std::uint64_t last_task_id,
    std::uint64_t first_ticket, std::uint64_t last_ticket,
    std::uint64_t logical_tasks, std::uint64_t logical_yields,
    std::uint64_t participant_hash) noexcept;
[[nodiscard]] std::uint64_t
JoinBatchOrderHash(std::uint64_t task_id, std::uint64_t target_task_id,
                   ReasonCode code, std::uint64_t logical_count) noexcept;
[[nodiscard]] std::uint64_t RootSingleJoinEpochOrderHash(
    std::uint64_t first_task_id, std::uint64_t last_task_id,
    std::uint64_t parent_task_id, std::uint64_t scope_id,
    std::uint64_t logical_tasks, std::uint64_t logical_events,
    std::uint64_t spawn_participant_hash) noexcept;

} // namespace rund::node

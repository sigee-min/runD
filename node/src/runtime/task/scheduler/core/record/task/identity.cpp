#include "local.hpp"

namespace rund::node {

[[nodiscard]] std::uint64_t Avalanche(std::uint64_t value) noexcept {
  value ^= value >> 30u;
  value *= 0xbf58476d1ce4e5b9ull;
  value ^= value >> 27u;
  value *= 0x94d049bb133111ebull;
  value ^= value >> 31u;
  return value;
}

[[nodiscard]] std::uint64_t
CanonicalOrderHash(const ::rund::detail::task::OperationKind kind,
                   const std::uint64_t a, const std::uint64_t b,
                   const std::uint64_t c, const std::uint64_t d,
                   const std::uint64_t e) noexcept {
  std::uint64_t hash = kFnvOffset;
  hash ^= static_cast<std::uint64_t>(kind) * 0x9e3779b97f4a7c15ull;
  hash ^= (a + 0xd1b54a32d192ed03ull) * 0x94d049bb133111ebull;
  hash ^= (b + 0x8a5cd789635d2dffull) * 0xbf58476d1ce4e5b9ull;
  hash ^= (c + 0x0123456789abcdefull) * 0xd6e8feb86659fd93ull;
  hash ^= (d + 0xfedcba9876543210ull) * 0xa0761d6478bd642full;
  hash ^= (e + 0xe7037ed1a0b428dbull) * 0xe7037ed1a0b428dbull;
  return Avalanche(hash);
}

[[nodiscard]] std::uint64_t SpawnBatchOrderHash(
    const std::uint64_t first_task_id, const std::uint64_t last_task_id,
    const std::uint64_t parent_task_id, const std::uint64_t scope_id,
    const std::uint64_t logical_count,
    const std::uint64_t participant_hash) noexcept {
  std::uint64_t hash = CanonicalOrderHash(
      ::rund::detail::task::OperationKind::TaskSpawnBatch, first_task_id,
      last_task_id, parent_task_id, scope_id, logical_count);
  MixHash(hash, participant_hash);
  return hash;
}

[[nodiscard]] std::uint64_t AppendSpawnParticipantHash(
    std::uint64_t participant_hash, const std::uint64_t task_id,
    const std::uint64_t parent_task_id, const std::uint64_t scope_id,
    const std::uint64_t name_hash) noexcept {
  MixHash(participant_hash,
          CanonicalOrderHash(::rund::detail::task::OperationKind::Spawn,
                             task_id, parent_task_id, scope_id, name_hash));
  return participant_hash;
}

[[nodiscard]] std::uint64_t
TerminalBatchOrderHash(const bool includes_root_submit,
                       const ::rund::detail::task::OperationKind terminal_kind,
                       const ReasonCode code, const std::uint64_t task_id,
                       const std::uint64_t logical_count) noexcept {
  return CanonicalOrderHash(
      ::rund::detail::task::OperationKind::TaskTerminalBatch,
      includes_root_submit ? 1u : 0u, static_cast<std::uint64_t>(terminal_kind),
      static_cast<std::uint64_t>(code), task_id, logical_count);
}

[[nodiscard]] std::uint64_t TerminalRangeBatchOrderHash(
    const bool includes_root_submit,
    const ::rund::detail::task::OperationKind terminal_kind,
    const ReasonCode code, const std::uint64_t first_task_id,
    const std::uint64_t last_task_id, const std::uint64_t first_ticket,
    const std::uint64_t last_ticket, const std::uint64_t logical_tasks,
    const std::uint64_t participant_hash) noexcept {
  std::uint64_t hash = CanonicalOrderHash(
      ::rund::detail::task::OperationKind::TaskTerminalBatch,
      includes_root_submit ? 1u : 0u, static_cast<std::uint64_t>(terminal_kind),
      static_cast<std::uint64_t>(code), first_task_id, last_task_id);
  MixHash(hash, first_ticket);
  MixHash(hash, last_ticket);
  MixHash(hash, logical_tasks);
  MixHash(hash, participant_hash);
  return hash;
}

[[nodiscard]] std::uint64_t
YieldBatchOrderHash(const bool includes_root_submit,
                    const std::uint64_t task_id,
                    const std::uint64_t logical_count,
                    const std::uint64_t logical_yields) noexcept {
  return CanonicalOrderHash(::rund::detail::task::OperationKind::YieldBatch,
                            includes_root_submit ? 1u : 0u, task_id,
                            logical_count, logical_yields);
}

[[nodiscard]] std::uint64_t LaneLocalYieldEpochOrderHash(
    const std::uint64_t first_task_id, const std::uint64_t last_task_id,
    const std::uint64_t first_ticket, const std::uint64_t last_ticket,
    const std::uint64_t logical_tasks, const std::uint64_t logical_yields,
    const std::uint64_t participant_hash) noexcept {
  std::uint64_t hash = CanonicalOrderHash(
      ::rund::detail::task::OperationKind::YieldBatch, first_task_id,
      last_task_id, first_ticket, last_ticket, logical_tasks);
  MixHash(hash, logical_yields);
  MixHash(hash, participant_hash);
  return hash;
}

[[nodiscard]] std::uint64_t
JoinBatchOrderHash(const std::uint64_t task_id,
                   const std::uint64_t target_task_id, const ReasonCode code,
                   const std::uint64_t logical_count) noexcept {
  return CanonicalOrderHash(::rund::detail::task::OperationKind::JoinBatch,
                            task_id, target_task_id,
                            static_cast<std::uint64_t>(code), logical_count);
}

[[nodiscard]] std::uint64_t RootSingleJoinEpochOrderHash(
    const std::uint64_t first_task_id, const std::uint64_t last_task_id,
    const std::uint64_t parent_task_id, const std::uint64_t scope_id,
    const std::uint64_t logical_tasks, const std::uint64_t logical_events,
    const std::uint64_t spawn_participant_hash) noexcept {
  std::uint64_t hash = CanonicalOrderHash(
      ::rund::detail::task::OperationKind::TaskRootJoinEpochBatch,
      first_task_id, last_task_id, parent_task_id, scope_id, logical_tasks);
  MixHash(hash, logical_events);
  MixHash(hash, spawn_participant_hash);
  MixHash(hash, static_cast<std::uint64_t>(
                    ::rund::detail::task::OperationKind::Complete));
  MixHash(hash, static_cast<std::uint64_t>(ReasonCode::Ok));
  return hash;
}

} // namespace rund::node

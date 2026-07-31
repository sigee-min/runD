#include <rund/task/stats/slots.hpp>

#include "../../state/storage.hpp"
#include "../../state/storage/check.hpp"

namespace rund::node {
namespace {

enum class IdentityDomain : std::uint8_t { Raw, Task, Scope };

[[nodiscard]] IdentityDomain
TargetDomain(const ::rund::detail::task::OperationKind kind) noexcept {
  switch (kind) {
  case ::rund::detail::task::OperationKind::ScopeEnter:
  case ::rund::detail::task::OperationKind::ScopePark:
  case ::rund::detail::task::OperationKind::ScopeWake:
    return IdentityDomain::Scope;
  case ::rund::detail::task::OperationKind::None:
  case ::rund::detail::task::OperationKind::RootSubmit:
  case ::rund::detail::task::OperationKind::Spawn:
  case ::rund::detail::task::OperationKind::Complete:
  case ::rund::detail::task::OperationKind::Fail:
  case ::rund::detail::task::OperationKind::Yield:
  case ::rund::detail::task::OperationKind::SleepZero:
  case ::rund::detail::task::OperationKind::TimerPark:
  case ::rund::detail::task::OperationKind::TimerWake:
  case ::rund::detail::task::OperationKind::JoinPark:
  case ::rund::detail::task::OperationKind::JoinWake:
  case ::rund::detail::task::OperationKind::ChannelMake:
  case ::rund::detail::task::OperationKind::ChannelSend:
  case ::rund::detail::task::OperationKind::ChannelRecv:
  case ::rund::detail::task::OperationKind::ChannelClose:
  case ::rund::detail::task::OperationKind::ChannelWake:
  case ::rund::detail::task::OperationKind::IoPark:
  case ::rund::detail::task::OperationKind::IoWake:
  case ::rund::detail::task::OperationKind::DeadlockWake:
  case ::rund::detail::task::OperationKind::ChannelMatch:
  case ::rund::detail::task::OperationKind::ChannelMatchBatch:
  case ::rund::detail::task::OperationKind::TaskSpawnBatch:
  case ::rund::detail::task::OperationKind::TaskTerminalBatch:
  case ::rund::detail::task::OperationKind::YieldBatch:
  case ::rund::detail::task::OperationKind::JoinBatch:
  case ::rund::detail::task::OperationKind::PrimitiveTrap:
  case ::rund::detail::task::OperationKind::TaskRootJoinEpochBatch:
  case ::rund::detail::task::OperationKind::ExternalPark:
  case ::rund::detail::task::OperationKind::ExternalWake:
    return IdentityDomain::Task;
  }
  return IdentityDomain::Raw;
}

[[nodiscard]] IdentityDomain
RegionDomain(const ::rund::detail::task::OperationKind kind) noexcept {
  switch (kind) {
  case ::rund::detail::task::OperationKind::TaskSpawnBatch:
    return IdentityDomain::Scope;
  case ::rund::detail::task::OperationKind::TaskTerminalBatch:
    return IdentityDomain::Task;
  case ::rund::detail::task::OperationKind::None:
  case ::rund::detail::task::OperationKind::RootSubmit:
  case ::rund::detail::task::OperationKind::Spawn:
  case ::rund::detail::task::OperationKind::Complete:
  case ::rund::detail::task::OperationKind::Fail:
  case ::rund::detail::task::OperationKind::Yield:
  case ::rund::detail::task::OperationKind::SleepZero:
  case ::rund::detail::task::OperationKind::TimerPark:
  case ::rund::detail::task::OperationKind::TimerWake:
  case ::rund::detail::task::OperationKind::JoinPark:
  case ::rund::detail::task::OperationKind::JoinWake:
  case ::rund::detail::task::OperationKind::ScopeEnter:
  case ::rund::detail::task::OperationKind::ScopePark:
  case ::rund::detail::task::OperationKind::ScopeWake:
  case ::rund::detail::task::OperationKind::ChannelMake:
  case ::rund::detail::task::OperationKind::ChannelSend:
  case ::rund::detail::task::OperationKind::ChannelRecv:
  case ::rund::detail::task::OperationKind::ChannelClose:
  case ::rund::detail::task::OperationKind::ChannelWake:
  case ::rund::detail::task::OperationKind::IoPark:
  case ::rund::detail::task::OperationKind::IoWake:
  case ::rund::detail::task::OperationKind::DeadlockWake:
  case ::rund::detail::task::OperationKind::ChannelMatch:
  case ::rund::detail::task::OperationKind::ChannelMatchBatch:
  case ::rund::detail::task::OperationKind::YieldBatch:
  case ::rund::detail::task::OperationKind::JoinBatch:
  case ::rund::detail::task::OperationKind::PrimitiveTrap:
  case ::rund::detail::task::OperationKind::TaskRootJoinEpochBatch:
  case ::rund::detail::task::OperationKind::ExternalPark:
  case ::rund::detail::task::OperationKind::ExternalWake:
    return IdentityDomain::Raw;
  }
  return IdentityDomain::Raw;
}

[[nodiscard]] std::uint64_t Project(SchedulerPlanState &plan,
                                    const IdentityDomain domain,
                                    const std::uint64_t value) noexcept {
  switch (domain) {
  case IdentityDomain::Raw:
    return value;
  case IdentityDomain::Task:
    return plan.task(value);
  case IdentityDomain::Scope:
    return plan.scope(value);
  }
  plan.fail();
  return 0u;
}

} // namespace

void Scheduler::RecordPhysical(
    const ::rund::detail::task::OperationKind kind, const ReasonCode code,
    const std::uint64_t task_id, const std::uint64_t target_id,
    const std::uint64_t wait_id, const std::uint64_t channel_id, const int fd,
    const short interest, const short revents, const std::int64_t deadline_ns,
    const std::uint64_t value_count, const std::uint64_t match_sequence,
    const std::uint64_t baton_epoch, const std::uint64_t task_op_ordinal,
    const std::uint64_t target_op_ordinal, const std::uint64_t region_id,
    const std::uint64_t epoch_id, const std::uint64_t logical_count,
    const std::uint64_t order_hash, const ReasonCode side_exit_code) noexcept {
  std::lock_guard lock{state_->evidence.mutex};
  state_->RequireSequencer();
  const bool canonical = state_->plan.installed;
  if (canonical && state_->plan.failure != ReasonCode::Ok) {
    return;
  }
  const std::uint64_t canonical_task =
      canonical ? state_->plan.task(task_id) : task_id;
  const std::uint64_t canonical_target =
      canonical ? Project(state_->plan, TargetDomain(kind), target_id)
                : target_id;
  const std::uint64_t canonical_wait =
      canonical ? state_->plan.wait(wait_id) : wait_id;
  const std::uint64_t canonical_channel =
      canonical ? state_->plan.channel(channel_id) : channel_id;
  const bool ticket_fields =
      kind == ::rund::detail::task::OperationKind::YieldBatch ||
      kind == ::rund::detail::task::OperationKind::TaskTerminalBatch;
  const std::uint64_t canonical_match =
      canonical && ticket_fields ? state_->plan.ticket(match_sequence)
                                 : match_sequence;
  const std::uint64_t canonical_baton = canonical && ticket_fields
                                            ? state_->plan.ticket(baton_epoch)
                                            : baton_epoch;
  const bool parent_task =
      kind == ::rund::detail::task::OperationKind::TaskSpawnBatch ||
      kind == ::rund::detail::task::OperationKind::TaskRootJoinEpochBatch;
  const std::uint64_t canonical_task_ordinal =
      canonical && parent_task ? state_->plan.task(task_op_ordinal)
                               : task_op_ordinal;
  const std::uint64_t canonical_region =
      canonical ? Project(state_->plan, RegionDomain(kind), region_id)
                : region_id;
  const std::uint64_t canonical_epoch =
      canonical &&
              kind ==
                  ::rund::detail::task::OperationKind::TaskRootJoinEpochBatch
          ? state_->plan.trace_epoch(epoch_id)
          : epoch_id;
  const std::uint64_t canonical_value =
      canonical && kind == ::rund::detail::task::OperationKind::JoinBatch
          ? state_->plan.task(value_count)
          : value_count;
  const int canonical_fd = canonical ? state_->plan.descriptor(fd) : fd;
  if (canonical && state_->plan.failure != ReasonCode::Ok) {
    return;
  }
  if (state_->evidence.input_capture_active.load(std::memory_order_relaxed)) {
    state_->evidence.input_capture.mutated = true;
  }
  auto &metrics = state_->evidence.metrics;
  auto &sequence = ::rund::detail::task::Stat(
      metrics, ::rund::detail::task::StatSlot::Operations);
  ++sequence;
  HashOperation(metrics, sequence, kind, canonical_task, canonical_target,
                canonical_wait, canonical_channel, canonical_fd, interest,
                revents, deadline_ns, canonical_value, canonical_match,
                canonical_baton, canonical_task_ordinal, target_op_ordinal,
                canonical_region, canonical_epoch, logical_count, order_hash,
                side_exit_code, code);
}

void Scheduler::Record(
    const ::rund::detail::task::OperationKind kind, const ReasonCode code,
    const std::uint64_t task_id, const std::uint64_t target_id,
    const std::uint64_t wait_id, const std::uint64_t channel_id, const int fd,
    const short interest, const short revents, const std::int64_t deadline_ns,
    const std::uint64_t value_count, const std::uint64_t match_sequence,
    const std::uint64_t baton_epoch, const std::uint64_t task_op_ordinal,
    const std::uint64_t target_op_ordinal, const std::uint64_t region_id,
    const std::uint64_t epoch_id, const std::uint64_t logical_count,
    const std::uint64_t order_hash, const ReasonCode side_exit_code) noexcept {
  std::lock_guard lock{state_->evidence.mutex};
  state_->RequireSequencer();
  if (kind != ::rund::detail::task::OperationKind::TaskRootJoinEpochBatch &&
      state_->batches.root_single_join_epoch.active) {
    FlushRootSingleJoinEpoch(ReasonCode::Ok);
  }
  if (state_->batches.task_spawn_batch.active) {
    FlushTaskSpawnBatch(ReasonCode::Ok);
  }
  if (kind != ::rund::detail::task::OperationKind::RootSubmit) {
    if (state_->batches.yield_resume_batch.active) {
      FlushYieldBatch(ReasonCode::Ok);
    }
    SchedulerThreadContext *const context = active_scheduler_context;
    if (context != nullptr && context->scheduler == this &&
        context->pending_root_submit) {
      FlushPendingRootSubmit();
    }
  }
  RecordPhysical(kind, code, task_id, target_id, wait_id, channel_id, fd,
                 interest, revents, deadline_ns, value_count, match_sequence,
                 baton_epoch, task_op_ordinal, target_op_ordinal, region_id,
                 epoch_id, logical_count, order_hash, side_exit_code);
}

} // namespace rund::node

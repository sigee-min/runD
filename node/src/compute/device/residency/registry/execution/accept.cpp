#include "../../execution/plan.hpp"
#include "../internal.hpp"

#include "../../registry/execution_owner.hpp"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <mutex>

namespace rund::compute::detail::residency {

bool ExecutionOwner::accept_execution_window(
    const execution::Plan &plan,
    const execution::WindowEvidence &evidence) noexcept {
  std::lock_guard lock{authority_.gate_};
  if (authority_.view_commit_quarantined_locked() ||
      authority_.execution_state_.slot.token == 0u ||
      authority_.execution_state_.slot.epochs == 0u ||
      authority_.execution_state_.slot.sliding_admitted ||
      authority_.execution_state_.slot.plan != plan.identity() ||
      authority_.execution_state_.slot.epochs != plan.epoch_count() ||
      evidence.plan_identity != authority_.execution_state_.slot.plan ||
      evidence.token != authority_.execution_state_.slot.token ||
      evidence.generation != authority_.execution_state_.slot.generation ||
      evidence.epoch_count != authority_.execution_state_.slot.epochs ||
      evidence.public_handoffs != 1u ||
      (evidence.native_batches != 0u && evidence.queue_calls == 0u) ||
      evidence.release_count == 0u ||
      evidence.release_count >
          authority_.execution_state_.slot.releases.size() ||
      evidence.first_epoch !=
          authority_.execution_state_.slot.window_accept_count ||
      evidence.first_epoch > authority_.execution_state_.slot.epochs ||
      evidence.release_count >
          authority_.execution_state_.slot.epochs - evidence.first_epoch ||
      authority_.execution_state_.slot.release_count !=
          evidence.first_epoch + evidence.release_count ||
      evidence.release_count !=
          std::min<std::uint64_t>(execution::WindowCapacity,
                                  authority_.execution_state_.slot.epochs -
                                      evidence.first_epoch) ||
      (evidence.native_batches != 0u &&
       (evidence.native_inflight_peak == 0u ||
        evidence.native_inflight_peak > execution::BankCapacity)) ||
      (evidence.native_batches == 0u && evidence.native_inflight_peak != 0u) ||
      evidence.native_batches > evidence.release_count ||
      evidence.queue_calls > evidence.native_batches ||
      authority_.execution_state_.slot.window_final ||
      authority_.execution_state_.slot.native_accepted ||
      authority_.execution_state_.slot.native_rejected ||
      evidence.terminal != (authority_.execution_state_.slot.unknown
                                ? execution::TerminalKind::UnknownMayWrite
                                : execution::TerminalKind::Known) ||
      static_cast<bool>(evidence.status) !=
          !authority_.execution_state_.slot.failed ||
      (evidence.status &&
       evidence.first_epoch + evidence.release_count !=
           authority_.execution_state_.slot.epochs &&
       authority_.execution_state_.slot.failed)) {
    return false;
  }
  std::uint64_t native_batches = 0u;
  for (std::size_t index = 0u; index < evidence.release_count; ++index) {
    const execution::Release &left = evidence.releases[index];
    const std::uint64_t epoch = evidence.first_epoch + index;
    const std::size_t slot =
        static_cast<std::size_t>(epoch % execution::WindowCapacity);
    if (authority_.execution_state_.slot.release_epochs[slot] != epoch) {
      return false;
    }
    const execution::Release &right =
        authority_.execution_state_.slot.releases[slot];
    if (left.status.reason() != right.status.reason() ||
        left.terminal != right.terminal ||
        left.plan_identity != right.plan_identity ||
        left.token != right.token || left.generation != right.generation ||
        left.epoch != right.epoch ||
        left.backend_sequence != right.backend_sequence ||
        left.bank != right.bank || left.dispatched != right.dispatched ||
        left.completed != right.completed ||
        left.may_write != right.may_write) {
      return false;
    }
    native_batches += left.dispatched ? 1u : 0u;
  }
  if (native_batches != evidence.native_batches) {
    return false;
  }
  authority_.execution_state_.slot.progress.native_inflight_peak =
      std::max(authority_.execution_state_.slot.progress.native_inflight_peak,
               evidence.native_inflight_peak);
  authority_.execution_state_.slot.window_accept_count +=
      evidence.release_count;
  authority_.execution_state_.slot.window_final =
      authority_.execution_state_.slot.failed ||
      authority_.execution_state_.slot.unknown ||
      authority_.execution_state_.slot.window_accept_count ==
          authority_.execution_state_.slot.epochs;
  return true;
}

bool ExecutionOwner::accept_execution_schedule(
    const execution::Plan &plan,
    const execution::ScheduleEvidence &evidence) noexcept {
  std::lock_guard lock{authority_.gate_};
  if (authority_.view_commit_quarantined_locked()) {
    return false;
  }
  const bool unknown =
      evidence.terminal == execution::TerminalKind::UnknownMayWrite &&
      !evidence.status && authority_.execution_state_.slot.unknown;
  const bool release_extent =
      unknown ? (authority_.execution_state_.slot.release_count != 0u &&
                 authority_.execution_state_.slot.release_count <=
                     authority_.execution_state_.slot.epochs)
              : authority_.execution_state_.slot.release_count ==
                    authority_.execution_state_.slot.epochs;
  if (authority_.execution_state_.slot.token == 0u ||
      authority_.execution_state_.slot.epochs == 0u ||
      authority_.execution_state_.slot.sliding_admitted ||
      authority_.execution_state_.slot.plan != plan.identity() ||
      authority_.execution_state_.slot.epochs != plan.epoch_count() ||
      evidence.plan_identity != authority_.execution_state_.slot.plan ||
      evidence.token != authority_.execution_state_.slot.token ||
      evidence.generation != authority_.execution_state_.slot.generation ||
      evidence.epoch_count != authority_.execution_state_.slot.epochs ||
      evidence.public_handoffs != 1u ||
      evidence.native_batches != authority_.execution_state_.slot.epochs ||
      evidence.queue_calls == 0u ||
      evidence.queue_calls > evidence.native_batches ||
      evidence.native_inflight_peak == 0u ||
      evidence.native_inflight_peak > execution::BankCapacity ||
      evidence.released_prefix !=
          authority_.execution_state_.slot.release_count ||
      !release_extent ||
      (!unknown && authority_.execution_state_.slot.native_inflight != 0u) ||
      authority_.execution_state_.slot.window_final ||
      authority_.execution_state_.slot.native_accepted ||
      authority_.execution_state_.slot.native_rejected ||
      evidence.terminal != (authority_.execution_state_.slot.unknown
                                ? execution::TerminalKind::UnknownMayWrite
                                : execution::TerminalKind::Known) ||
      static_cast<bool>(evidence.status) !=
          !authority_.execution_state_.slot.failed ||
      evidence.completed_ns == 0u ||
      (!unknown &&
       (authority_.execution_state_.slot.progress.native_dispatches !=
            authority_.execution_state_.slot.epochs ||
        authority_.execution_state_.slot.progress.native_completions !=
            authority_.execution_state_.slot.epochs)) ||
      (evidence.status &&
       (authority_.execution_state_.slot.progress.input_services !=
            authority_.execution_state_.slot.epochs ||
        authority_.execution_state_.slot.progress.output_services !=
            authority_.execution_state_.slot.epochs))) {
    return false;
  }
  authority_.execution_state_.slot.progress.native_inflight_peak =
      std::max(authority_.execution_state_.slot.progress.native_inflight_peak,
               evidence.native_inflight_peak);
  // The native Final authenticates that all Q batches were accepted by the
  // one handoff. UnknownMayWrite may have only a strict terminal prefix; count
  // the remaining accepted batches without fabricating completion receipts.
  authority_.execution_state_.slot.progress.native_dispatches =
      evidence.native_batches;
  authority_.execution_state_.slot.window_accept_count =
      authority_.execution_state_.slot.release_count;
  authority_.execution_state_.slot.window_final = true;
  return true;
}

bool ExecutionOwner::accept_execution(
    const execution::Plan &plan,
    const execution::NativeEvidence &evidence) noexcept {
  std::lock_guard lock{authority_.gate_};
  if (authority_.view_commit_quarantined_locked()) {
    return false;
  }
  constexpr std::uint8_t DispatchMask =
      std::uint8_t{1u} << static_cast<std::uint8_t>(execution::Phase::Dispatch);
  if (authority_.execution_state_.slot.token == 0u ||
      authority_.execution_state_.slot.epochs != 1u ||
      authority_.execution_state_.slot.sliding_admitted ||
      plan.identity() != authority_.execution_state_.slot.plan ||
      plan.epoch_count() != 1u ||
      evidence.plan_identity != authority_.execution_state_.slot.plan ||
      evidence.token != authority_.execution_state_.slot.token ||
      evidence.generation != authority_.execution_state_.slot.generation ||
      evidence.epoch_count != authority_.execution_state_.slot.epochs ||
      evidence.native_submissions != 1u ||
      authority_.execution_state_.slot.native_accepted ||
      authority_.execution_state_.slot.native_rejected ||
      authority_.execution_state_.slot.failed ||
      authority_.execution_state_.slot.progress.input_services != 1u ||
      authority_.execution_state_.slot.progress.output_services != 0u ||
      authority_.execution_state_.slot.next_sequence != 2u ||
      evidence.native_dispatches > 1u ||
      evidence.native_completions > evidence.native_dispatches ||
      evidence.native_inflight_peak > 1u ||
      (evidence.native_dispatches != 0u &&
       evidence.native_inflight_peak == 0u) ||
      evidence.failure_count > evidence.failures.size() ||
      (evidence.terminal == execution::TerminalKind::UnknownMayWrite &&
       evidence.status) ||
      (evidence.status &&
       (evidence.terminal != execution::TerminalKind::Known ||
        evidence.native_dispatches != 1u || evidence.native_completions != 1u ||
        evidence.native_inflight_peak != 1u || evidence.failure_count != 0u)) ||
      (!evidence.status && evidence.failure_count == 0u)) {
    return false;
  }
  if (evidence.native_dispatches != 1u) {
    return false;
  }
  for (std::size_t index = 0u; index < evidence.failure_count; ++index) {
    const execution::FailureEvidence failure = evidence.failures[index];
    if (failure.epoch != 0u || failure.phases == 0u ||
        (failure.phases & ~DispatchMask) != 0u ||
        (failure.may_write & ~DispatchMask) != 0u ||
        (failure.terminal & ~DispatchMask) != 0u ||
        (failure.phases & ~failure.may_write) != 0u ||
        (evidence.terminal == execution::TerminalKind::Known &&
         (failure.may_write & ~failure.terminal) != 0u) ||
        index != 0u) {
      return false;
    }
    const std::uint64_t terminal =
        (failure.terminal & DispatchMask) != 0u ? 1u : 0u;
    if (evidence.native_completions != terminal) {
      return false;
    }
  }

  if (evidence.status && authority_.execution_state_.slot.cache_admitted) {
    execution::Node dispatch{};
    if (!plan.project(
            execution::NodeId{.epoch = 0u, .phase = execution::Phase::Dispatch},
            dispatch)) {
      return false;
    }
    for (std::size_t local = 0u; local < dispatch.output_count; ++local) {
      const CacheUse &use = dispatch.output[local];
      const std::uint32_t frame =
          dispatch.route.target.first + static_cast<std::uint32_t>(local);
      const ExecutionFrame prior = authority_.frames_[frame];
      if ((prior.state != FrameState::Empty &&
           prior.state != FrameState::Resident) ||
          !prior.dirty.empty()) {
        return false;
      }
      authority_.frames_[frame] =
          ExecutionFrame{.key = use.key,
                         .next_use = use.next_use,
                         .retain_until = use.retain_until,
                         .state = FrameState::Pinned,
                         .tier = prior.tier,
                         .role = prior.role,
                         .dirty = use.dirty,
                         .extent = prior.extent,
                         .view = prior.view,
                         .assigned = true};
    }
  }

  authority_.execution_state_.slot.native = evidence;
  authority_.execution_state_.slot.native_accepted = true;
  authority_.execution_state_.slot.progress.native_dispatches =
      evidence.native_dispatches;
  authority_.execution_state_.slot.progress.native_completions =
      evidence.native_completions;
  authority_.execution_state_.slot.progress.native_inflight_peak =
      evidence.native_inflight_peak;
  authority_.execution_state_.slot.native_inflight =
      evidence.native_dispatches - evidence.native_completions;
  if (evidence.native_dispatches != 0u) {
    const std::size_t phase =
        static_cast<std::size_t>(execution::Phase::Dispatch);
    authority_.execution_state_.slot.issued[0u][phase] = 0u;
    authority_.execution_state_.slot.terminals[0u][phase] =
        evidence.native_completions == 1u ? 0u : NeverUse;
    authority_.execution_state_.slot.sequences[0u][phase] =
        authority_.execution_state_.slot.next_sequence;
    authority_.execution_state_.slot.may_write[0u][phase] = true;
  }
  ++authority_.execution_state_.slot.next_sequence;
  authority_.execution_state_.slot.failed = !evidence.status;
  authority_.execution_state_.slot.unknown =
      evidence.terminal == execution::TerminalKind::UnknownMayWrite;
  return true;
}

} // namespace rund::compute::detail::residency

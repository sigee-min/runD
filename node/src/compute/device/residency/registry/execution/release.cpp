#include "../../execution/plan.hpp"
#include "../internal.hpp"

#include "../../registry/execution_owner.hpp"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <mutex>

namespace rund::compute::detail::residency {

bool ExecutionOwner::release_execution(const std::uint64_t token,
                                       const std::uint64_t generation,
                                       const execution::Plan &plan,
                                       const execution::Release &release,
                                       const std::uint64_t sequence) noexcept {
  std::lock_guard lock{authority_.gate_};
  if (authority_.view_commit_quarantined_locked()) {
    return false;
  }
  constexpr std::uint8_t DispatchMask =
      std::uint8_t{1u} << static_cast<std::uint8_t>(execution::Phase::Dispatch);
  if (authority_.execution_state_.slot.token == 0u ||
      authority_.execution_state_.slot.epochs == 0u || token == 0u ||
      authority_.execution_state_.slot.sliding_admitted || generation == 0u ||
      authority_.execution_state_.slot.token != token ||
      authority_.execution_state_.slot.generation != generation ||
      authority_.execution_state_.slot.plan != plan.identity() ||
      authority_.execution_state_.slot.epochs != plan.epoch_count() ||
      release.plan_identity != authority_.execution_state_.slot.plan ||
      release.token != token || release.generation != generation ||
      release.epoch >= authority_.execution_state_.slot.epochs ||
      release.epoch != authority_.execution_state_.slot.release_count ||
      release.bank != release.epoch % execution::BankCapacity ||
      release.backend_sequence !=
          authority_.execution_state_.slot.release_count + 1u ||
      sequence == 0u ||
      sequence != authority_.execution_state_.slot.next_sequence ||
      authority_.execution_state_.slot.native_accepted ||
      authority_.execution_state_.slot.native_rejected ||
      authority_.execution_state_.slot.window_final ||
      (release.dispatched &&
       authority_.execution_state_.slot.release_count != 0u &&
       (authority_.execution_state_.slot.release_epochs
                [(authority_.execution_state_.slot.release_count - 1u) %
                 execution::WindowCapacity] !=
            authority_.execution_state_.slot.release_count - 1u ||
        !authority_.execution_state_.slot
             .releases[(authority_.execution_state_.slot.release_count - 1u) %
                       execution::WindowCapacity]
             .dispatched)) ||
      (release.dispatched && (release.status || release.may_write) &&
       !(authority_.execution_state_.slot.unknown &&
         release.terminal == execution::TerminalKind::UnknownMayWrite) &&
       authority_.execution_state_.slot.progress.input_services <=
           release.epoch) ||
      (release.dispatched && (release.status || release.may_write) &&
       !(authority_.execution_state_.slot.unknown &&
         release.terminal == execution::TerminalKind::UnknownMayWrite) &&
       release.epoch >= execution::BankCapacity &&
       authority_.execution_state_.slot.progress.output_services <=
           release.epoch - execution::BankCapacity) ||
      (release.completed && !release.dispatched) ||
      (release.may_write && !release.dispatched) ||
      (release.terminal == execution::TerminalKind::UnknownMayWrite &&
       (!release.dispatched || release.completed || !release.may_write ||
        release.status)) ||
      (release.status &&
       (release.terminal != execution::TerminalKind::Known ||
        !release.dispatched || !release.completed || !release.may_write)) ||
      (!release.status && release.terminal == execution::TerminalKind::Known &&
       ((release.dispatched && !release.completed) ||
        (!release.dispatched && (release.completed || release.may_write))))) {
    return false;
  }

  const std::size_t bank = release.bank;
  const std::size_t phase =
      static_cast<std::size_t>(execution::Phase::Dispatch);
  const bool collapsed_unknown =
      authority_.execution_state_.slot.unknown && !release.status &&
      release.terminal == execution::TerminalKind::UnknownMayWrite &&
      release.dispatched && !release.completed && release.may_write;
  if (!collapsed_unknown && release.dispatched &&
      authority_.execution_state_.slot.issued[bank][phase] != NeverUse &&
      authority_.execution_state_.slot.terminals[bank][phase] !=
          authority_.execution_state_.slot.issued[bank][phase]) {
    return false;
  }
  // Validate every fact that can fail before mutating progress or frame
  // metadata. This keeps a malformed adapter Release a fail-closed no-op.
  if (!collapsed_unknown &&
      authority_.execution_state_.slot.window_cache_admitted &&
      (release.status || release.may_write)) {
    const std::size_t input_service = 0u;
    if (authority_.execution_state_.slot
                .window_service_epoch[input_service][bank] != release.epoch ||
        authority_.execution_state_.slot
                .window_service_binding_count[input_service][bank] == 0u) {
      return false;
    }
    for (std::size_t index = 0u;
         index < authority_.execution_state_.slot
                     .window_service_binding_count[input_service][bank];
         ++index) {
      const CacheBinding &binding =
          authority_.execution_state_.slot
              .window_service_bindings[input_service][bank][index];
      const std::size_t input_count =
          authority_.execution_state_.slot
              .window_service_binding_count[input_service][bank] /
          2u;
      if (index < input_count &&
          (authority_.execution_state_.slot
               .window_service_coherent_mask[input_service][bank] &
           (std::uint32_t{1u} << index)) != 0u) {
        continue;
      }
      if (binding.frame >= authority_.frames_.size() ||
          authority_.frames_[binding.frame].key != binding.key ||
          authority_.frames_[binding.frame].state != FrameState::Pinned) {
        return false;
      }
    }
    if (release.status) {
      execution::Node dispatch{};
      if (!plan.project(execution::NodeId{.epoch = release.epoch,
                                          .phase = execution::Phase::Dispatch},
                        dispatch) ||
          dispatch.bank != bank || dispatch.output_count == 0u) {
        return false;
      }
      for (std::size_t local = 0u; local < dispatch.output_count; ++local) {
        const std::uint32_t frame =
            dispatch.route.target.first + static_cast<std::uint32_t>(local);
        if (frame >= authority_.frames_.size() ||
            (authority_.frames_[frame].state != FrameState::Empty &&
             authority_.frames_[frame].state != FrameState::Resident) ||
            !authority_.frames_[frame].dirty.empty()) {
          return false;
        }
      }
    }
  }
  if (release.dispatched) {
    authority_.execution_state_.slot.issued[bank][phase] = release.epoch;
    authority_.execution_state_.slot.terminals[bank][phase] =
        release.completed ? release.epoch : NeverUse;
    authority_.execution_state_.slot.sequences[bank][phase] = sequence;
    authority_.execution_state_.slot.may_write[bank][phase] = release.may_write;
  }
  authority_.execution_state_.slot.progress.native_dispatches +=
      release.dispatched ? 1u : 0u;
  authority_.execution_state_.slot.progress.native_completions +=
      release.completed ? 1u : 0u;
  authority_.execution_state_.slot.progress.native_inflight_peak =
      std::max<std::uint64_t>(
          authority_.execution_state_.slot.progress.native_inflight_peak,
          release.dispatched && !release.completed ? 1u : 0u);
  authority_.execution_state_.slot.native_inflight +=
      release.dispatched && !release.completed ? 1u : 0u;

  if (!collapsed_unknown &&
      authority_.execution_state_.slot.window_cache_admitted &&
      (release.status || release.may_write)) {
    const std::size_t input_service = 0u;
    if (release.status) {
      execution::Node dispatch{};
      static_cast<void>(
          plan.project(execution::NodeId{.epoch = release.epoch,
                                         .phase = execution::Phase::Dispatch},
                       dispatch));
      for (std::size_t local = 0u; local < dispatch.output_count; ++local) {
        const CacheUse &use = dispatch.output[local];
        const std::uint32_t frame =
            dispatch.route.target.first + static_cast<std::uint32_t>(local);
        const ExecutionFrame prior = authority_.frames_[frame];
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
    for (std::size_t index = 0u;
         index < authority_.execution_state_.slot
                     .window_service_binding_count[input_service][bank];
         ++index) {
      const CacheBinding &binding =
          authority_.execution_state_.slot
              .window_service_bindings[input_service][bank][index];
      const std::size_t input_count =
          authority_.execution_state_.slot
              .window_service_binding_count[input_service][bank] /
          2u;
      if (index < input_count &&
          (authority_.execution_state_.slot
               .window_service_coherent_mask[input_service][bank] &
           (std::uint32_t{1u} << index)) != 0u) {
        continue;
      }
      authority_.frames_[binding.frame].dirty = {};
      authority_.frames_[binding.frame].state = FrameState::Resident;
    }
    authority_.execution_state_.slot.window_service_epoch[input_service][bank] =
        NeverUse;
    authority_.execution_state_.slot
        .window_service_binding_count[input_service][bank] = 0u;
    authority_.execution_state_.slot
        .window_service_transition_count[input_service][bank] = 0u;
    authority_.execution_state_.slot
        .window_service_backing_mask[input_service][bank] = 0u;
    authority_.execution_state_.slot
        .window_service_transfer_mask[input_service][bank] = 0u;
    authority_.execution_state_.slot
        .window_service_coherent_mask[input_service][bank] = 0u;
  }
  const std::size_t release_slot =
      authority_.execution_state_.slot.release_count %
      execution::WindowCapacity;
  authority_.execution_state_.slot.releases[release_slot] = release;
  authority_.execution_state_.slot.release_epochs[release_slot] = release.epoch;
  ++authority_.execution_state_.slot.release_count;
  ++authority_.execution_state_.slot.next_sequence;
  if (release.status) {
    return true;
  }
  authority_.execution_state_.slot.failed = true;
  authority_.execution_state_.slot.unknown =
      authority_.execution_state_.slot.unknown ||
      release.terminal == execution::TerminalKind::UnknownMayWrite;
  const bool known_no_write =
      release.terminal == execution::TerminalKind::Known &&
      release.dispatched && release.completed && !release.may_write;
  // An upfront schedule may drain an arbitrary accepted suffix through
  // identical no-write gates after the first exact failure. Those rows do not
  // name additional mutations. Retain at most the first such cause instead
  // of overflowing the bounded mutation-failure journal into a false Unknown.
  if (known_no_write && authority_.execution_state_.slot.failure_count != 0u) {
    return true;
  }
  if (authority_.execution_state_.slot.failure_count ==
      authority_.execution_state_.slot.failures.size()) {
    authority_.execution_state_.slot.unknown = true;
    return true;
  }
  authority_.execution_state_.slot
      .failures[authority_.execution_state_.slot.failure_count++] =
      ExecutionFailure{
          .epoch = release.epoch,
          .phases = DispatchMask,
          .may_write = release.may_write ? DispatchMask : std::uint8_t{0u},
          .terminal = release.terminal == execution::TerminalKind::Known
                          ? DispatchMask
                          : std::uint8_t{0u},
      };
  return true;
}

} // namespace rund::compute::detail::residency

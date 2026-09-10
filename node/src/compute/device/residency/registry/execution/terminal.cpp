#include "../../registry/execution_owner.hpp"
#include "../internal.hpp"
#include "../lease_state.hpp"

#include "../../execution/plan.hpp"
#include "../../execution/sliding.hpp"
#include "../frame.hpp"

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <mutex>

namespace rund::compute::detail::residency {

bool ExecutionOwner::terminal_execution(
    const ExecutionTicket &ticket, const ExecutionTerminal terminal) noexcept {
  std::lock_guard lock{authority_.gate_};
  if (authority_.view_commit_quarantined_locked()) {
    return false;
  }
  const std::size_t bank = ticket.bank;
  const std::size_t phase = ticket.phase;
  if (ticket.token == 0u || ticket.generation == 0u ||
      authority_.execution_state_.slot.sliding_admitted ||
      authority_.execution_state_.slot.token != ticket.token ||
      authority_.execution_state_.slot.generation != ticket.generation ||
      authority_.execution_state_.slot.plan != ticket.plan ||
      bank >= execution::BankCapacity || phase >= 3u ||
      authority_.execution_state_.slot.issued[bank][phase] != ticket.epoch ||
      authority_.execution_state_.slot.terminals[bank][phase] != NeverUse ||
      authority_.execution_state_.slot.sequences[bank][phase] !=
          ticket.sequence ||
      authority_.execution_state_.slot.may_write[bank][phase] !=
          ticket.may_write) {
    return false;
  }
  const bool service =
      authority_.execution_state_.slot.cache_admitted && ticket.epoch == 0u &&
      (phase == static_cast<std::size_t>(execution::Phase::Input) ||
       phase == static_cast<std::size_t>(execution::Phase::Output));
  const bool window_service =
      authority_.execution_state_.slot.window_cache_admitted &&
      (phase == static_cast<std::size_t>(execution::Phase::Input) ||
       phase == static_cast<std::size_t>(execution::Phase::Output));
  if (service) {
    const std::size_t service_index =
        phase == static_cast<std::size_t>(execution::Phase::Output) ? 1u : 0u;
    if (ticket.bindings.data() !=
            authority_.execution_state_.slot.service_bindings[service_index]
                .data() ||
        ticket.bindings.size() != authority_.execution_state_.slot
                                      .service_binding_count[service_index] ||
        ticket.transitions.data() !=
            authority_.execution_state_.slot.service_transitions[service_index]
                .data() ||
        ticket.transitions.size() !=
            authority_.execution_state_.slot
                .service_transition_count[service_index] ||
        ticket.backing_mask != authority_.execution_state_.slot
                                   .service_backing_mask[service_index] ||
        ticket.transfer_mask != authority_.execution_state_.slot
                                    .service_transfer_mask[service_index] ||
        ticket.coherent_mask != 0u) {
      return false;
    }
  } else if (window_service) {
    const std::size_t service_index =
        phase == static_cast<std::size_t>(execution::Phase::Output) ? 1u : 0u;
    if (authority_.execution_state_.slot
                .window_service_epoch[service_index][bank] != ticket.epoch ||
        ticket.bindings.data() !=
            authority_.execution_state_.slot
                .window_service_bindings[service_index][bank]
                .data() ||
        ticket.bindings.size() !=
            authority_.execution_state_.slot
                .window_service_binding_count[service_index][bank] ||
        ticket.transitions.data() !=
            authority_.execution_state_.slot
                .window_service_transitions[service_index][bank]
                .data() ||
        ticket.transitions.size() !=
            authority_.execution_state_.slot
                .window_service_transition_count[service_index][bank] ||
        ticket.backing_mask !=
            authority_.execution_state_.slot
                .window_service_backing_mask[service_index][bank] ||
        ticket.transfer_mask !=
            authority_.execution_state_.slot
                .window_service_transfer_mask[service_index][bank] ||
        ticket.coherent_mask !=
            authority_.execution_state_.slot
                .window_service_coherent_mask[service_index][bank]) {
      return false;
    }
  } else if (!ticket.bindings.empty() || !ticket.transitions.empty() ||
             ticket.backing_mask != 0u || ticket.transfer_mask != 0u ||
             ticket.coherent_mask != 0u) {
    return false;
  }
  if (terminal == ExecutionTerminal::Success && (service || window_service)) {
    const std::size_t input_count =
        window_service &&
                phase == static_cast<std::size_t>(execution::Phase::Input)
            ? ticket.bindings.size() / 2u
            : 0u;
    for (std::size_t index = 0u; index < ticket.bindings.size(); ++index) {
      const CacheBinding &binding = ticket.bindings[index];
      if (index < input_count &&
          (ticket.coherent_mask & (std::uint32_t{1u} << index)) != 0u) {
        continue;
      }
      if (binding.frame >= authority_.frames_.size() ||
          authority_.frames_[binding.frame].key != binding.key ||
          (authority_.frames_[binding.frame].state != FrameState::Mapping &&
           authority_.frames_[binding.frame].state != FrameState::Pinned)) {
        return false;
      }
    }
  }
  if (phase == static_cast<std::size_t>(execution::Phase::Dispatch) &&
      authority_.execution_state_.slot.native_inflight == 0u) {
    return false;
  }
  authority_.execution_state_.slot.terminals[bank][phase] = ticket.epoch;
  if (phase == static_cast<std::size_t>(execution::Phase::Input)) {
    ++authority_.execution_state_.slot.progress.input_services;
  } else if (phase == static_cast<std::size_t>(execution::Phase::Dispatch)) {
    --authority_.execution_state_.slot.native_inflight;
    ++authority_.execution_state_.slot.progress.native_completions;
  } else {
    ++authority_.execution_state_.slot.progress.output_services;
  }
  if (terminal == ExecutionTerminal::Success) {
    if (service || window_service) {
      const std::size_t input_count =
          window_service &&
                  phase == static_cast<std::size_t>(execution::Phase::Input)
              ? ticket.bindings.size() / 2u
              : 0u;
      for (std::size_t index = 0u; index < ticket.bindings.size(); ++index) {
        const CacheBinding &binding = ticket.bindings[index];
        if (index < input_count &&
            (ticket.coherent_mask & (std::uint32_t{1u} << index)) != 0u) {
          continue;
        }
        authority_.frames_[binding.frame].state = FrameState::Pinned;
      }
    }
    if (window_service &&
        phase == static_cast<std::size_t>(execution::Phase::Output)) {
      // Output service success means both the physical supply and the backing
      // stage named by this exact ticket completed. Neither output owner is a
      // cache authority after persistence; retire both before this bank can be
      // dispatched again.
      for (std::size_t local = 0u; local < ticket.bindings.size() / 2u;
           ++local) {
        const std::uint32_t source = ticket.bindings[local * 2u].frame;
        const std::uint32_t target = ticket.bindings[local * 2u + 1u].frame;
        authority_.frames_[source] =
            frame_detail::empty(authority_.frames_[source]);
        authority_.frames_[target] =
            frame_detail::empty(authority_.frames_[target]);
      }
      authority_.execution_state_.slot.window_service_epoch[1u][bank] =
          NeverUse;
      authority_.execution_state_.slot.window_service_binding_count[1u][bank] =
          0u;
      authority_.execution_state_.slot
          .window_service_transition_count[1u][bank] = 0u;
      authority_.execution_state_.slot.window_service_backing_mask[1u][bank] =
          0u;
      authority_.execution_state_.slot.window_service_transfer_mask[1u][bank] =
          0u;
    }
    return true;
  }
  authority_.execution_state_.slot.failed = true;
  authority_.execution_state_.slot.unknown =
      authority_.execution_state_.slot.unknown ||
      terminal == ExecutionTerminal::UnknownMayWrite;
  if (window_service &&
      phase == static_cast<std::size_t>(execution::Phase::Output)) {
    authority_.execution_state_.slot.window_service_epoch[1u][bank] = NeverUse;
    authority_.execution_state_.slot.window_service_binding_count[1u][bank] =
        0u;
    authority_.execution_state_.slot.window_service_transition_count[1u][bank] =
        0u;
    authority_.execution_state_.slot.window_service_backing_mask[1u][bank] = 0u;
    authority_.execution_state_.slot.window_service_transfer_mask[1u][bank] =
        0u;
  }
  const std::uint8_t mask = static_cast<std::uint8_t>(1u << phase);
  for (std::size_t index = 0u;
       index < authority_.execution_state_.slot.failure_count; ++index) {
    if (authority_.execution_state_.slot.failures[index].epoch ==
        ticket.epoch) {
      authority_.execution_state_.slot.failures[index].phases =
          static_cast<std::uint8_t>(
              authority_.execution_state_.slot.failures[index].phases | mask);
      authority_.execution_state_.slot.failures[index].may_write =
          static_cast<std::uint8_t>(
              authority_.execution_state_.slot.failures[index].may_write |
              (ticket.may_write ? mask : std::uint8_t{0u}));
      authority_.execution_state_.slot.failures[index].terminal =
          static_cast<std::uint8_t>(
              authority_.execution_state_.slot.failures[index].terminal | mask);
      return true;
    }
  }
  if (authority_.execution_state_.slot.failure_count ==
      authority_.execution_state_.slot.failures.size()) {
    authority_.execution_state_.slot.unknown = true;
    return true;
  }
  authority_.execution_state_.slot
      .failures[authority_.execution_state_.slot.failure_count++] =
      ExecutionFailure{
          .epoch = ticket.epoch,
          .phases = mask,
          .may_write = ticket.may_write ? mask : std::uint8_t{0u},
          .terminal = mask,
      };
  return true;
}

} // namespace rund::compute::detail::residency

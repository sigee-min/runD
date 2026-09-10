#include "../../../registry/execution_owner.hpp"

#include "../../../execution/plan.hpp"
#include "../../../execution/sliding.hpp"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <limits>

namespace rund::compute::detail::residency {

void ExecutionOwner::finalize_execution_issue_locked(
    const std::uint64_t token, const std::uint64_t generation,
    const execution::Plan &plan, const execution::Node &node,
    const std::uint64_t sequence, const std::size_t bank,
    const std::size_t phase, const bool window_service,
    ExecutionTicket &ticket) noexcept {
  authority_.execution_state_.slot.issued[bank][phase] = node.id.epoch;
  authority_.execution_state_.slot.terminals[bank][phase] = NeverUse;
  authority_.execution_state_.slot.sequences[bank][phase] = sequence;
  authority_.execution_state_.slot.may_write[bank][phase] = node.may_write;
  if (node.id.phase == execution::Phase::Dispatch) {
    ++authority_.execution_state_.slot.progress.native_dispatches;
    ++authority_.execution_state_.slot.native_inflight;
    authority_.execution_state_.slot.progress.native_inflight_peak =
        std::max(authority_.execution_state_.slot.progress.native_inflight_peak,
                 authority_.execution_state_.slot.native_inflight);
  }
  ++authority_.execution_state_.slot.next_sequence;
  const bool service = authority_.execution_state_.slot.cache_admitted &&
                       node.id.epoch == 0u &&
                       node.domain == execution::Domain::HostService;
  const std::size_t service_index =
      node.id.phase == execution::Phase::Output ? 1u : 0u;
  ticket = ExecutionTicket{
      .token = token,
      .generation = generation,
      .plan = plan.identity(),
      .epoch = node.id.epoch,
      .sequence = sequence,
      .phase = static_cast<std::uint8_t>(phase),
      .bank = static_cast<std::uint8_t>(bank),
      .bindings =
          service ? std::span<
                        const CacheBinding>{authority_.execution_state_.slot
                                                .service_bindings[service_index]
                                                .data(),
                                            authority_.execution_state_.slot
                                                .service_binding_count
                                                    [service_index]}
          : window_service
              ? std::span<const CacheBinding>{authority_.execution_state_.slot
                                                  .window_service_bindings
                                                      [service_index][bank]
                                                  .data(),
                                              authority_.execution_state_.slot
                                                  .window_service_binding_count
                                                      [service_index][bank]}
              : std::span<const CacheBinding>{},
      .transitions =
          service ? std::span<
                        const CacheTransition>{authority_.execution_state_.slot
                                                   .service_transitions
                                                       [service_index]
                                                   .data(),
                                               authority_.execution_state_.slot
                                                   .service_transition_count
                                                       [service_index]}
          : window_service
              ? std::span<
                    const CacheTransition>{authority_.execution_state_.slot
                                               .window_service_transitions
                                                   [service_index][bank]
                                               .data(),
                                           authority_.execution_state_.slot
                                               .window_service_transition_count
                                                   [service_index][bank]}
              : std::span<const CacheTransition>{},
      .backing_mask =
          service ? authority_.execution_state_.slot
                        .service_backing_mask[service_index]
          : window_service
              ? authority_.execution_state_.slot
                    .window_service_backing_mask[service_index][bank]
              : 0u,
      .transfer_mask =
          service ? authority_.execution_state_.slot
                        .service_transfer_mask[service_index]
          : window_service
              ? authority_.execution_state_.slot
                    .window_service_transfer_mask[service_index][bank]
              : 0u,
      .coherent_mask =
          window_service
              ? authority_.execution_state_.slot
                    .window_service_coherent_mask[service_index][bank]
              : 0u,
      .may_write = node.may_write,
  };
}

} // namespace rund::compute::detail::residency

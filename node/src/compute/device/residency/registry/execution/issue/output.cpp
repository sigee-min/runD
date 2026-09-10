#include "../../../registry/execution_owner.hpp"

#include "../../../execution/plan.hpp"
#include "../../../execution/sliding.hpp"
#include "../../frame.hpp"
#include "../../internal.hpp"
#include "../../lease_state.hpp"

#include <cstddef>
#include <cstdint>

namespace rund::compute::detail::residency {

bool ExecutionOwner::issue_execution_output_service_locked(
    const execution::Node &node) noexcept {
  if (!(authority_.execution_state_.slot.cache_admitted &&
        node.id.epoch == 0u && node.id.phase == execution::Phase::Output)) {
    return true;
  }
  if (!authority_.execution_state_.slot.native_accepted ||
      !authority_.execution_state_.slot.native.status ||
      authority_.execution_state_.slot.output_admitted ||
      node.output_count > execution::UseCapacity) {
    return false;
  }
  for (std::size_t local = 0u; local < node.output_count; ++local) {
    const CacheUse &use = node.output[local];
    const std::uint32_t source =
        node.route.source.first + static_cast<std::uint32_t>(local);
    const std::uint32_t target =
        node.route.target.first + static_cast<std::uint32_t>(local);
    const ExecutionFrame &source_frame = authority_.frames_[source];
    const ExecutionFrame prior = authority_.frames_[target];
    if (source_frame.key != use.key ||
        source_frame.state != FrameState::Pinned ||
        source_frame.dirty != use.dirty ||
        (prior.state != FrameState::Empty &&
         prior.state != FrameState::Resident) ||
        !prior.dirty.empty()) {
      return false;
    }
    const std::size_t source_binding =
        authority_.execution_state_.slot.service_binding_count[1u]++;
    authority_.execution_state_.slot.service_bindings[1u][source_binding] =
        CacheBinding{
            .key = use.key,
            .frame = source,
            .access = Access::Write,
            .dirty = use.dirty,
            .prior_dirty = source_frame.dirty,
            .next_use = use.next_use,
            .retain_until = use.retain_until,
        };
    if (prior.state != FrameState::Empty) {
      const std::size_t index =
          authority_.execution_state_.slot.service_transition_count[1u]++;
      authority_.execution_state_.slot.service_transitions[1u][index] =
          CacheTransition{
              .key = prior.key,
              .frame = target,
              .kind = TransitionKind::Unmap,
          };
    }
    {
      const std::size_t index =
          authority_.execution_state_.slot.service_transition_count[1u]++;
      authority_.execution_state_.slot.service_transitions[1u][index] =
          CacheTransition{
              .key = use.key,
              .frame = target,
              .kind = TransitionKind::Map,
          };
    }
    {
      const std::size_t index =
          authority_.execution_state_.slot.service_transition_count[1u]++;
      authority_.execution_state_.slot.service_transitions[1u][index] =
          CacheTransition{
              .key = use.key,
              .frame = target,
              .kind = TransitionKind::Writeback,
              .dirty = use.dirty,
          };
    }
    authority_.frames_[target] =
        ExecutionFrame{.key = use.key,
                       .next_use = use.next_use,
                       .retain_until = use.retain_until,
                       .state = FrameState::Mapping,
                       .tier = prior.tier,
                       .role = prior.role,
                       .dirty = use.dirty,
                       .extent = prior.extent,
                       .view = prior.view,
                       .assigned = true};
    const std::size_t target_binding =
        authority_.execution_state_.slot.service_binding_count[1u]++;
    authority_.execution_state_.slot.service_bindings[1u][target_binding] =
        CacheBinding{
            .key = use.key,
            .frame = target,
            .access = Access::Write,
            .dirty = use.dirty,
            .prior_dirty = prior.dirty,
            .next_use = use.next_use,
            .retain_until = use.retain_until,
        };
    const std::uint32_t bit = std::uint32_t{1u} << local;
    authority_.execution_state_.slot.service_transfer_mask[1u] |= bit;
    authority_.execution_state_.slot.service_backing_mask[1u] |= bit;
  }
  authority_.execution_state_.slot.output_admitted = true;
  return true;
}

} // namespace rund::compute::detail::residency

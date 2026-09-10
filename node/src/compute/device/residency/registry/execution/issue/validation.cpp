#include "../../../registry/execution_owner.hpp"

#include "../../../execution/plan.hpp"
#include "../../../execution/sliding.hpp"
#include "../../frame.hpp"
#include "../../internal.hpp"
#include "../../lease_state.hpp"
#include "../../release_check.hpp"

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <limits>

namespace rund::compute::detail::residency {

namespace {

[[nodiscard]] bool same(const CacheUse &left, const CacheUse &right) noexcept {
  return left.key == right.key && left.access == right.access &&
         left.next_use == right.next_use && left.dirty == right.dirty &&
         left.epoch == right.epoch && left.retain_until == right.retain_until;
}

[[nodiscard]] bool same(const execution::Node &left,
                        const execution::Node &right) noexcept {
  if (left.id != right.id || left.domain != right.domain ||
      left.bank != right.bank || left.route.source != right.route.source ||
      left.route.target != right.route.target ||
      left.input_count != right.input_count ||
      left.output_count != right.output_count ||
      left.mutation_count != right.mutation_count ||
      left.active_mask != right.active_mask ||
      left.may_write != right.may_write) {
    return false;
  }
  for (std::size_t index = 0u; index < left.input_count; ++index) {
    if (!same(left.input[index], right.input[index])) {
      return false;
    }
  }
  for (std::size_t index = 0u; index < left.output_count; ++index) {
    if (!same(left.output[index], right.output[index])) {
      return false;
    }
  }
  for (std::size_t index = 0u; index < left.mutation_count; ++index) {
    if (left.mutations[index] != right.mutations[index]) {
      return false;
    }
  }
  return true;
}

} // namespace

bool ExecutionOwner::validate_execution_issue_locked(
    const std::uint64_t token, const std::uint64_t generation,
    const execution::Plan &plan, const execution::Node &node,
    const std::uint64_t sequence, std::size_t &bank,
    std::size_t &phase) noexcept {
  if (authority_.view_commit_quarantined_locked()) {
    return false;
  }
  const std::size_t cleanup_slot =
      static_cast<std::size_t>(node.id.epoch % execution::WindowCapacity);
  const bool failed_output_cleanup =
      authority_.execution_state_.slot.failed &&
      node.id.phase == execution::Phase::Output &&
      node.id.epoch < authority_.execution_state_.slot.release_count &&
      authority_.execution_state_.slot.release_epochs[cleanup_slot] ==
          node.id.epoch &&
      authority_.execution_state_.slot.releases[cleanup_slot].status &&
      authority_.execution_state_.slot.releases[cleanup_slot].completed;
  if (token == 0u || generation == 0u ||
      authority_.execution_state_.slot.token != token ||
      authority_.execution_state_.slot.sliding_admitted ||
      authority_.execution_state_.slot.generation != generation ||
      (authority_.execution_state_.slot.failed && !failed_output_cleanup) ||
      plan.identity() != authority_.execution_state_.slot.plan ||
      plan.epoch_count() != authority_.execution_state_.slot.epochs ||
      sequence == 0u ||
      sequence != authority_.execution_state_.slot.next_sequence) {
    return false;
  }
  execution::Node projected{};
  if (!plan.project(node.id, projected) || !same(node, projected) ||
      node.bank >= execution::BankCapacity) {
    return false;
  }
  bank = node.bank;
  phase = static_cast<std::size_t>(node.id.phase);
  if (phase >= 3u ||
      (authority_.execution_state_.slot.issued[bank][phase] != NeverUse &&
       authority_.execution_state_.slot.terminals[bank][phase] !=
           authority_.execution_state_.slot.issued[bank][phase])) {
    return false;
  }
  std::array<execution::NodeId, execution::PredecessorCapacity> storage{};
  std::size_t predecessor_count = 0u;
  if (!plan.predecessors(node.id, storage, predecessor_count)) {
    return false;
  }
  for (std::size_t index = 0u; index < predecessor_count; ++index) {
    const execution::NodeId predecessor = storage[index];
    const std::size_t predecessor_bank =
        static_cast<std::size_t>(predecessor.epoch % execution::BankCapacity);
    const std::size_t predecessor_phase =
        static_cast<std::size_t>(predecessor.phase);
    if (authority_.execution_state_.slot
            .terminals[predecessor_bank][predecessor_phase] !=
        predecessor.epoch) {
      return false;
    }
  }
  return true;
}

} // namespace rund::compute::detail::residency

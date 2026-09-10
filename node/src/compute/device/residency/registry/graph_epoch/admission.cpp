#include "../../registry.hpp"
#include "../../registry/graph_persist_owner.hpp"
#include "../internal.hpp"
#include "../lease_state.hpp"
#include "internal.hpp"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <mutex>
#include <span>

namespace rund::compute::detail::residency {

AuthorityResult
Authority::begin_graph_epoch(const std::span<const PageUse> uses,
                             const std::span<const GraphPortRequest> ports,
                             const std::size_t anchor_port,
                             const std::uint64_t epoch,
                             const CpuReservationKey cpu_key) noexcept {
  std::lock_guard lock{gate_};
  return begin_graph_epoch_locked(uses, ports, anchor_port, epoch, cpu_key,
                                  nullptr);
}

AuthorityResult GraphPersistOwner::begin_cpu_graph_epoch_retry(
    const std::span<const PageUse> uses,
    const std::span<const GraphPortRequest> ports,
    const std::size_t anchor_port, const std::uint64_t epoch,
    const CpuReservationKey cpu_key,
    const GraphPersistIdentity &identity) noexcept {
  std::lock_guard lock{authority_.gate_};
  return authority_.begin_graph_epoch_locked(uses, ports, anchor_port, epoch,
                                             cpu_key, &identity);
}

AuthorityResult Authority::begin_graph_epoch_locked(
    const std::span<const PageUse> uses,
    const std::span<const GraphPortRequest> ports,
    const std::size_t anchor_port, const std::uint64_t epoch,
    const CpuReservationKey cpu_key,
    const GraphPersistIdentity *const retry_identity) noexcept {
  if (view_commit_quarantined_locked() || view_commit_inflight_locked()) {
    return AuthorityResult{.failure = AuthorityFailure::Busy};
  }
  if (retry_identity == nullptr && retry_ready(cycle_state_.graph_persists)) {
    return AuthorityResult{.failure = AuthorityFailure::Busy};
  }
  if (retry_identity != nullptr && !retry_identity->valid()) {
    return AuthorityResult{.failure = AuthorityFailure::Invalid};
  }
  if (supplied_cpu_key(cpu_key) &&
      (!valid_cpu_key(cpu_key) || cpu_key.owner() != credentials_.owner_id ||
       cpu_graph_state_.pending_cpu != cpu_key)) {
    return AuthorityResult{.failure = AuthorityFailure::Invalid};
  }
  if (!supplied_cpu_key(cpu_key) && cpu_graph_state_.pending_cpu) {
    return AuthorityResult{.failure = AuthorityFailure::Busy};
  }
  registry_model::PendingCpu pending{this, cpu_key, supplied_cpu_key(cpu_key)};

  graph_epoch_detail::AdmissionDraft draft{};
  const AuthorityResult request_check =
      graph_epoch_detail::Validation::requests(uses, ports, anchor_port, epoch,
                                               draft);
  if (!request_check) {
    return request_check;
  }

  if (execution_state_.slot.token != 0u) {
    return AuthorityResult{.failure = AuthorityFailure::Busy};
  }
  if (std::any_of(cycle_state_.epochs.begin(), cycle_state_.epochs.end(),
                  [](const LeaseSlot &slot) {
                    return slot.state == LeaseState::Prepared;
                  })) {
    return AuthorityResult{.failure = AuthorityFailure::Busy};
  }
  if (supplied_cpu_key(cpu_key) &&
      std::any_of(cycle_state_.epochs.begin(), cycle_state_.epochs.end(),
                  [cpu_key](const LeaseSlot &slot) {
                    return slot.state != LeaseState::Free &&
                           slot.cpu_key == cpu_key;
                  })) {
    return AuthorityResult{.failure = AuthorityFailure::Busy};
  }
  const AuthorityResult region_check =
      graph_epoch_detail::Validation::regions(*this, ports);
  if (!region_check) {
    return region_check;
  }
  const auto free = std::find_if(
      cycle_state_.epochs.begin(), cycle_state_.epochs.end(),
      [](const LeaseSlot &slot) { return slot.state == LeaseState::Free; });
  if (free == cycle_state_.epochs.end()) {
    return AuthorityResult{.failure = AuthorityFailure::Busy};
  }

  registry_model::RetryTxn retry{*this};
  if (retry_identity != nullptr &&
      !retry.stage(cpu_key ? cpu_key.domain() : 0u, *retry_identity)) {
    return AuthorityResult{.failure = AuthorityFailure::Busy};
  }
  const AuthorityResult assignment =
      graph_epoch_detail::Assignment::run(*this, ports, epoch, draft);
  if (!assignment) {
    return assignment;
  }

  LeaseSlot &slot = *free;
  const AuthorityResult staged = graph_epoch_detail::Relocation::run(
      *this, uses, ports, epoch, draft, slot);
  if (!staged) {
    return staged;
  }
  slot.token = next_token(credentials_.next_token);
  if (slot.token == 0u) {
    rollback(frames_, slot, false);
    clear(slot);
    return AuthorityResult{.failure = AuthorityFailure::Capacity};
  }
  slot.generation = next_generation(credentials_.next_generation);
  if (slot.generation == 0u) {
    rollback(frames_, slot, false);
    clear(slot);
    return AuthorityResult{.failure = AuthorityFailure::Capacity};
  }
  slot.coordinate = epoch;
  slot.cpu_key = cpu_key;
  slot.cpu_bound = false;
  slot.state = LeaseState::Prepared;
  retry.commit();
  pending.publish();
  return AuthorityResult{.failure = AuthorityFailure::None,
                         .lease = EpochLease{.bindings = slot.bindings,
                                             .transitions = slot.transitions,
                                             .ports = slot.ports,
                                             .relocations = slot.relocations,
                                             .remaps = slot.remaps,
                                             .token = slot.token,
                                             .generation = slot.generation}};
}

} // namespace rund::compute::detail::residency

#include "internal.hpp"

#include "../../execution/graph_persist/internal.hpp"
#include "../../registry.hpp"
#include "../frame.hpp"

#include <exception>
#include <memory>
#include <mutex>
#include <utility>

namespace rund::compute::detail::residency {

void registry_model::CpuQuarantineOwner::commit_q(const Plan &plan) noexcept {
  for (std::size_t index = 0u; index < plan.epoch_count; ++index) {
    registry_model::LeaseSlot &epoch =
        authority_.cycle_state_.epochs[plan.epochs[index]];
    if (!complete_epoch(authority_.frames_, epoch, false, true)) {
      std::terminate();
    }
  }
  for (std::size_t index = 0u; index < plan.unknown_count; ++index) {
    registry_model::LeaseSlot &row =
        authority_.cycle_state_.graph_persists[plan.unknowns[index]];
    graph_persist_detail::rollback(authority_.frames_, row);
    for (const CacheTransition &transition : row.transitions) {
      authority_.frames_[transition.frame] =
          frame_detail::empty(authority_.frames_[transition.frame]);
    }
    ::rund::compute::detail::residency::clear(row);
  }
  for (std::size_t index = 0u; index < plan.retry_count; ++index) {
    registry_model::LeaseSlot &row =
        authority_.cycle_state_.graph_persists[plan.retries[index]];
    for (const CacheTransition &transition : row.transitions) {
      authority_.frames_[transition.frame] =
          frame_detail::empty(authority_.frames_[transition.frame]);
    }
    ::rund::compute::detail::residency::clear(row);
  }
  authority_.cpu_graph_state_.cpu_quarantine.reset();
}

void registry_model::CpuQuarantineOwner::scrub_q(
    graph_reduce::CpuReceiptBook &book, const BookSnapshot &snap) noexcept {
  for (std::size_t index = 0u; index < snap.epochs.size(); ++index) {
    const BookSnapshot::Epoch &old = snap.epochs[index];
    graph_reduce::CpuReceiptBook::Slot &item = book.slots_[index];
    if (item.state != old.state || item.authority != old.authority ||
        item.bank != old.bank ||
        static_cast<std::uint8_t>(item.role) != old.role ||
        item.key != old.key || item.token != old.token ||
        item.generation != old.generation || item.permit != old.permit ||
        item.handle != old.handle) {
      std::terminate();
    }
    if (old.state != graph_reduce::CpuReceiptBook::SlotState::Free) {
      book.release_permit(item);
      book.reset(item);
    } else if (item.handle != nullptr) {
      book.reset(item);
    }
  }
  for (std::size_t index = 0u; index < snap.unknowns.size(); ++index) {
    const BookSnapshot::Unknown &old = snap.unknowns[index];
    graph_reduce::CpuReceiptBook::UnknownCred &item = book.unknowns_[index];
    if (item.live != old.live || item.authority != old.authority ||
        item.token != old.token || item.generation != old.generation ||
        item.coordinate != old.coordinate || item.plan != old.plan ||
        item.identity != old.identity || item.region != old.region ||
        item.page_count != old.page_count || item.domain != old.domain ||
        item.terminal != old.terminal || item.quarantined != old.quarantined) {
      std::terminate();
    }
    if (item.live) {
      item.clear();
    }
  }
  if (book.live_receipts_ != 0u || book.live_permits_ != 0u) {
    std::terminate();
  }
}

bool registry_model::CpuQuarantineOwner::discard(
    graph_reduce::CpuReceiptBook &book,
    const void *const quarantine_identity) noexcept {
  const BookSnapshot snap = snap_q(book);
  std::unique_lock lock{authority_.gate_};
  if (authority_.view_commit_quarantined_locked() ||
      authority_.cpu_graph_state_.cpu_quarantine.get() != quarantine_identity ||
      authority_.cpu_graph_state_.pending_cpu ||
      authority_.execution_state_.slot.token != 0u ||
      authority_.execution_state_.slot.native_inflight != 0u ||
      authority_.cycle_state_.writeback.state !=
          registry_model::LeaseState::Free ||
      authority_.cycle_state_.cycle.token != 0u ||
      authority_.cycle_state_.cycle.count != 0u || !snap.valid ||
      snap.domain == 0u) {
    return false;
  }
  Plan plan{};
  if (!check_q(snap, plan)) {
    return false;
  }
  commit_q(plan);
  lock.unlock();
  scrub_q(book, snap);
  return true;
}

} // namespace rund::compute::detail::residency

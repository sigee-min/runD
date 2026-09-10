#include "../../receipts.hpp"

#include "../../../../../device/residency/registry.hpp"
#include "../../../../../device/residency/registry/cpu_graph_owner.hpp"
#include "../../../../../device/residency/registry/graph_persist_owner.hpp"

#include <algorithm>

namespace rund::compute::detail::graph_reduce {

bool CpuReceiptBook::can_hold_unknown(
    residency::Authority &authority,
    const residency::execution::GraphPersist &ticket) const noexcept {
  if (!ticket || !ticket.unknown() || ticket.book_domain() != domain_ ||
      ticket.coordinate() == 0u ||
      !authority.graph_persists().validate_cpu_graph_persist(ticket)) {
    return false;
  }
  return std::none_of(unknowns_.begin(), unknowns_.end(),
                      [&](const UnknownCred &slot) {
                        return slot.live && slot.token == ticket.token();
                      }) &&
         std::any_of(unknowns_.begin(), unknowns_.end(),
                     [](const UnknownCred &slot) { return !slot.live; });
}

bool CpuReceiptBook::hold_unknown(
    residency::Authority &authority,
    residency::execution::GraphPersist &ticket) noexcept {
  if (!can_hold_unknown(authority, ticket)) {
    return false;
  }
  const auto free =
      std::find_if(unknowns_.begin(), unknowns_.end(),
                   [](const UnknownCred &slot) { return !slot.live; });
  if (free == unknowns_.end()) {
    return false;
  }
  free->authority = &authority;
  free->plan = ticket.plan();
  free->identity = ticket.identity();
  free->region = ticket.region();
  free->domain = ticket.book_domain();
  free->token = ticket.token();
  free->generation = ticket.generation();
  free->coordinate = ticket.coordinate();
  free->page_count = ticket.pages().size();
  free->terminal = true;
  free->quarantined = true;
  free->live = true;
  ticket.clear();
  return true;
}

bool CpuReceiptBook::recover(residency::Authority &authority,
                             residency::CloseInfo *const info) noexcept {
  if (info != nullptr) {
    *info = residency::CloseInfo{};
  }
  bool clean = true;
  for (Slot &slot : slots_) {
    if (slot.state == SlotState::Free) {
      continue;
    }
    if (slot.state == SlotState::Reserved) {
      if (slot.token == 0u && slot.generation == 0u) {
        if (slot.authority != nullptr &&
            slot.authority->cpu_graph().cancel_cpu_reservation(slot.key)) {
          release_permit(slot);
          reset_cred(slot);
        } else {
          clean = false;
        }
        continue;
      }
      if (info != nullptr && info->check == residency::CloseInfo::Check::None) {
        info->check = residency::CloseInfo::Check::Credential;
        info->token = slot.token;
        info->generation = slot.generation;
      }
      clean = false;
      continue;
    }
    residency::CloseInfo local{};
    const bool owner = slot.authority == &authority;
    if (!owner) {
      local.check = residency::CloseInfo::Check::Credential;
      local.token = slot.token;
      local.generation = slot.generation;
    }
    if (!owner ||
        !authority.cpu_graph().close_cpu_epoch(
            slot.key, slot.token, slot.generation, false, true, &local)) {
      if (info != nullptr && info->check == residency::CloseInfo::Check::None) {
        *info = local;
      }
      clean = false;
      continue;
    }
    release_permit(slot);
    reset_cred(slot);
  }
  return clean;
}

} // namespace rund::compute::detail::graph_reduce

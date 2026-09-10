#pragma once

#include "capability.hpp"
#include "service.hpp"
#include "validation.hpp"

#include <cstddef>
#include <cstdint>
#include <memory>

namespace rund::node::accel::detail {

// Cold backend-owned lowering for one immutable persistent request.  The
// service table can only open/wait/ack already-submitted work; only its submit
// entry may create the single native queue submission.
struct PersistentResidencySlidingPreparation final {
  PersistentResidencySlidingCapability capability{};
  PersistentResidencySlidingServiceOps service{};
  std::shared_ptr<void> lowering{};
  std::shared_ptr<PersistentResidencySlidingRequestCell> cell{};
  std::shared_ptr<PersistentResidencySlidingTicket> ticket{};
  std::uint64_t encoded_coordinate_count{};
  std::uint64_t encoded_intermediate_bytes{};

  [[nodiscard]] PersistentResidencySlidingRequest
  active_request() const noexcept {
    if (ticket != nullptr && ticket->active && ticket->pending.has_value()) {
      PersistentResidencySlidingRequest result = *ticket->pending;
      result.ticket = ticket;
      return result;
    }
    if (cell != nullptr) {
      PersistentResidencySlidingRequest result = cell->committed;
      result.ticket = ticket;
      return result;
    }
    return {};
  }

  [[nodiscard]] bool pending_ticket() const noexcept {
    return ticket != nullptr && ticket->active && ticket->pending.has_value();
  }

  [[nodiscard]] explicit operator bool() const noexcept {
    PersistentResidencySlidingRequest request = active_request();
    request.lowering = lowering;
    return persistent_sliding_product_capable(capability) &&
           service.submit_result != nullptr &&
           service.wait_done != nullptr &&
           service.signal_ready != nullptr && service.ack_done != nullptr &&
           service.fail_service != nullptr &&
           service.quarantine_unknown != nullptr && lowering != nullptr &&
           cell != nullptr && ticket != nullptr && ticket->cell == cell &&
           cell->id != 0u && cell->domain != 0u &&
           persistent_sliding_request_valid(capability, request);
  }
};

[[nodiscard]] inline bool persistent_sliding_shape_equal(
    const PersistentResidencySlidingRequest &left,
    const PersistentResidencySlidingRequest &right) noexcept {
  if (left.plan_identity != right.plan_identity ||
      left.coordinate_count != right.coordinate_count ||
      left.cell_id != right.cell_id || left.cell_domain != right.cell_domain ||
      left.memory != right.memory || left.width != right.width ||
      left.mode != right.mode) {
    return false;
  }
  for (std::size_t slot = 0u; slot < left.width; ++slot) {
    const auto &a = left.roles[slot];
    const auto &b = right.roles[slot];
    if (a.slot != b.slot) {
      return false;
    }
  }
  return true;
}

[[nodiscard]] inline bool persistent_sliding_request_equal(
    const PersistentResidencySlidingRequest &left,
    const PersistentResidencySlidingRequest &right) noexcept {
  if (left.plan_identity != right.plan_identity || left.token != right.token ||
      left.generation != right.generation ||
      left.owner_nonce != right.owner_nonce || left.cell_id != right.cell_id ||
      left.cell_domain != right.cell_domain ||
      left.coordinate_count != right.coordinate_count ||
      left.tail_local_count != right.tail_local_count ||
      left.admission != right.admission || left.final != right.final ||
      left.user != right.user || left.memory != right.memory ||
      left.width != right.width || left.mode != right.mode) {
    return false;
  }
  for (std::size_t slot = 0u; slot < left.width; ++slot) {
    const auto &a = left.roles[slot];
    const auto &b = right.roles[slot];
    if (a.prepared != b.prepared || a.locals != b.locals ||
        a.local_count != b.local_count ||
        a.first_control_generation != b.first_control_generation ||
        a.control_generation_stride != b.control_generation_stride ||
        a.first_descriptor_generation != b.first_descriptor_generation ||
        a.descriptor_generation_stride != b.descriptor_generation_stride ||
        a.slot != b.slot) {
      return false;
    }
  }
  return true;
}

[[nodiscard]] inline bool persistent_sliding_stage_ticket(
    const std::shared_ptr<PersistentResidencySlidingTicket> &ticket,
    const PersistentResidencySlidingCapability &capability,
    const PersistentResidencySlidingRequest &next) noexcept {
  if (ticket == nullptr || ticket->cell == nullptr || ticket->quarantined ||
      !persistent_sliding_request_valid(capability, next) ||
      next.ticket.get() != ticket.get() || next.cell_id != ticket->cell->id ||
      next.cell_domain != ticket->cell->domain) {
    return false;
  }
  if (ticket->active && ticket->pending.has_value()) {
    return persistent_sliding_request_equal(*ticket->pending, next);
  }
  if (!persistent_sliding_shape_equal(ticket->cell->committed, next)) {
    return false;
  }
  ticket->pending = next;
  ticket->pending->lowering.reset();
  ticket->pending->ticket.reset();
  ticket->active = true;
  return true;
}

[[nodiscard]] inline bool persistent_sliding_commit_ticket(
    const std::shared_ptr<PersistentResidencySlidingTicket> &ticket,
    PersistentResidencySlidingRequest *const committed) noexcept {
  if (ticket == nullptr || ticket->cell == nullptr || !ticket->active ||
      !ticket->pending.has_value() || ticket->quarantined) {
    return false;
  }
  const PersistentResidencySlidingRequest next = *ticket->pending;
  ticket->cell->committed = next;
  ticket->cell->committed.lowering.reset();
  ticket->cell->committed.ticket.reset();
  if (committed != nullptr) {
    *committed = next;
    committed->lowering.reset();
  }
  ticket->pending.reset();
  ticket->active = false;
  return true;
}

inline void persistent_sliding_abort_ticket(
    const std::shared_ptr<PersistentResidencySlidingTicket> &ticket) noexcept {
  if (ticket == nullptr) {
    return;
  }
  ticket->pending.reset();
  ticket->active = false;
}

inline void persistent_sliding_quarantine_ticket(
    const std::shared_ptr<PersistentResidencySlidingTicket> &ticket) noexcept {
  if (ticket != nullptr) {
    ticket->quarantined = true;
  }
}

} // namespace rund::node::accel::detail

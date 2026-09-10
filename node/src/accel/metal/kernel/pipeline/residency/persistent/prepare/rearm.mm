#include "../internal.hpp"

#include <algorithm>
#include <limits>

namespace rund::node::accel::detail::metal_persistent_sliding {

#if defined(__APPLE__) && defined(RUND_NODE_HAVE_METAL_SDK)
namespace {

[[nodiscard]] bool
valid_coordinates(const Owner &owner,
                  const PersistentResidencySlidingRequest &request) noexcept {
  const std::size_t count = std::min<std::size_t>(
      SlotCount, static_cast<std::size_t>(request.coordinate_count));
  for (std::size_t index = 0u; index < count; ++index) {
    const Coordinate &entry = owner.coordinates[index];
    PersistentResidencySlidingServiceIdentity expected{};
    if (entry.sequence == nullptr ||
        !persistent_sliding_service_identity(request, index, expected)) {
      return false;
    }
  }
  return true;
}

void refresh_coordinates(
    Owner &owner, const PersistentResidencySlidingRequest &request) noexcept {
  const std::size_t count = std::min<std::size_t>(
      SlotCount, static_cast<std::size_t>(request.coordinate_count));
  for (std::size_t index = 0u; index < count; ++index) {
    Coordinate &entry = owner.coordinates[index];
    static_cast<void>(
        persistent_sliding_service_identity(request, index, entry.identity));
    entry.ready_value = owner.event_base + entry.identity.turn + 1u;
    entry.done_value = owner.event_base + index + 1u;
    entry.admission = {true, "ok"};
  }
  for (std::size_t index = count; index < SlotCount; ++index) {
    owner.coordinates[index] = {};
  }
}

} // namespace

bool coordinate_at(const Owner &owner, const std::uint64_t coordinate,
                   Coordinate &entry) noexcept {
  const PersistentResidencySlidingRequest &request = owner.prepared;
  if (request.width == 0u || coordinate >= request.coordinate_count) {
    return false;
  }
  if (!persistent_sliding_service_identity(request, coordinate,
                                           entry.identity)) {
    return false;
  }
  const std::size_t slot =
      static_cast<std::size_t>(coordinate % request.width);
  const PersistentResidencySlidingRole &role = request.roles[slot];
  entry.sequence = static_cast<MetalSequence *>(role.prepared.get());
  entry.locals = role.locals;
  entry.local_count = coordinate + 1u == request.coordinate_count
                          ? request.tail_local_count
                          : role.local_count;
  const std::uint64_t turn = coordinate / request.width;
  const std::uint64_t max = std::numeric_limits<std::uint64_t>::max();
  if (turn == max || coordinate == max ||
      owner.run_event_base > max - turn - 1u ||
      owner.run_event_base > max - coordinate - 1u) {
    return false;
  }
  entry.ready_value = owner.run_event_base + turn + 1u;
  entry.done_value = owner.run_event_base + coordinate + 1u;
  return entry.sequence != nullptr && entry.local_count != 0u &&
         entry.local_count <= entry.locals.size();
}

bool stage_rearm(Owner &owner,
                 const PersistentResidencySlidingRequest &request) noexcept {
  MetalAdapter *adapter = nullptr;
  if (owner.native_pending || owner.quarantine_ticket) {
    record_first_failure(
        owner, MetalPersistentResidencySlidingDiagnosticStage::StageRearm,
        request.coordinate_count,
        MetalPersistentResidencySlidingDiagnosticPredicate::State, 1u,
        PersistentSlidingNoCoordinate);
    return false;
  }
  if (request.owner_nonce == 0u || request.ticket == nullptr) {
    record_first_failure(
        owner, MetalPersistentResidencySlidingDiagnosticStage::StageRearm,
        request.coordinate_count,
        MetalPersistentResidencySlidingDiagnosticPredicate::Request, 1u,
        PersistentSlidingNoCoordinate);
    return false;
  }
  const std::uint64_t issue = first_invalid_request(request, adapter);
  if (issue != valid_request_issue()) {
    record_first_failure(
        owner, MetalPersistentResidencySlidingDiagnosticStage::StageRearm,
        request.coordinate_count,
        MetalPersistentResidencySlidingDiagnosticPredicate::Request, issue,
        PersistentSlidingNoCoordinate);
    return false;
  }
  if (adapter != owner.adapter) {
    record_first_failure(
        owner, MetalPersistentResidencySlidingDiagnosticStage::StageRearm,
        request.coordinate_count,
        MetalPersistentResidencySlidingDiagnosticPredicate::Request, 3u,
        PersistentSlidingNoCoordinate);
    return false;
  }
  if (!persistent_sliding_stage_ticket(request.ticket, owner.capability,
                                       request)) {
    record_first_failure(
        owner, MetalPersistentResidencySlidingDiagnosticStage::StageRearm,
        request.coordinate_count,
        MetalPersistentResidencySlidingDiagnosticPredicate::Ticket, 1u,
        PersistentSlidingNoCoordinate);
    return false;
  }
  if (!valid_coordinates(owner, request)) {
    record_first_failure(
        owner, MetalPersistentResidencySlidingDiagnosticStage::StageRearm,
        request.coordinate_count,
        MetalPersistentResidencySlidingDiagnosticPredicate::Coordinates,
        SlotCount, PersistentSlidingNoCoordinate);
    persistent_sliding_abort_ticket(request.ticket);
    return false;
  }
  owner.pending_request = request;
  owner.pending_request.lowering.reset();
  owner.native_pending = true;
  return true;
}

bool prepare_rearm(Owner &owner) noexcept {
  if (!owner.native_pending) {
    record_first_failure(
        owner, MetalPersistentResidencySlidingDiagnosticStage::Prepare, 0u,
        MetalPersistentResidencySlidingDiagnosticPredicate::State, 1u,
        PersistentSlidingNoCoordinate);
    return false;
  }
  const PersistentResidencySlidingRequest next = owner.pending_request;
  const PersistentResidencySlidingRequest old = owner.prepared;
  const id<MTLCommandBuffer> old_command = owner.command;
  if (next.coordinate_count >
      std::numeric_limits<std::uint64_t>::max() - owner.event_base) {
    record_first_failure(
        owner, MetalPersistentResidencySlidingDiagnosticStage::Prepare,
        next.coordinate_count,
        MetalPersistentResidencySlidingDiagnosticPredicate::Request,
        owner.event_base, PersistentSlidingNoCoordinate);
    return false;
  }
  owner.rollback_request = old;
  owner.rollback_command = old_command;
  owner.prepared = next;
  owner.prepared.lowering.reset();
  refresh_coordinates(owner, next);
  if (old.token == next.token && old.generation == next.generation) {
    owner.native_prepared = true;
    return true;
  }
  id<MTLCommandQueue> const queue =
      (__bridge id<MTLCommandQueue>)owner.adapter->queue.get();
  if (queue == nil) {
    record_first_failure(
        owner, MetalPersistentResidencySlidingDiagnosticStage::Encode,
        next.coordinate_count,
        MetalPersistentResidencySlidingDiagnosticPredicate::Command, 0u,
        PersistentSlidingNoCoordinate);
    owner.prepared = old;
    refresh_coordinates(owner, old);
    return false;
  }
  owner.command = [queue commandBuffer];
  if (owner.command == nil) {
    record_first_failure(
        owner, MetalPersistentResidencySlidingDiagnosticStage::Encode,
        next.coordinate_count,
        MetalPersistentResidencySlidingDiagnosticPredicate::Command, 1u,
        PersistentSlidingNoCoordinate);
    owner.command = old_command;
    owner.prepared = old;
    refresh_coordinates(owner, old);
    owner.rollback_request = {};
    owner.rollback_command = nil;
    return false;
  }
  if (!encode(owner).ok) {
    record_first_failure(
        owner, MetalPersistentResidencySlidingDiagnosticStage::Encode,
        next.coordinate_count,
        MetalPersistentResidencySlidingDiagnosticPredicate::Encode, 1u,
        PersistentSlidingNoCoordinate);
    owner.command = old_command;
    owner.prepared = old;
    refresh_coordinates(owner, old);
    owner.rollback_request = {};
    owner.rollback_command = nil;
    return false;
  }
  owner.native_prepared = true;
  return true;
}

bool commit_rearm(Owner &owner) noexcept {
  if (!owner.native_pending || !owner.native_prepared) {
    record_first_failure(
        owner, MetalPersistentResidencySlidingDiagnosticStage::CommonCommit, 0u,
        MetalPersistentResidencySlidingDiagnosticPredicate::State, 1u,
        PersistentSlidingNoCoordinate);
    return false;
  }
  const PersistentResidencySlidingRequest next = owner.pending_request;
  if (next.ticket == nullptr) {
    record_first_failure(
        owner, MetalPersistentResidencySlidingDiagnosticStage::CommonCommit,
        next.coordinate_count,
        MetalPersistentResidencySlidingDiagnosticPredicate::Ticket, 1u,
        PersistentSlidingNoCoordinate);
    owner.quarantine_ticket = true;
    persistent_sliding_quarantine_ticket(next.ticket);
    return false;
  }
  if (!persistent_sliding_commit_ticket(next.ticket, &owner.prepared)) {
    record_first_failure(
        owner, MetalPersistentResidencySlidingDiagnosticStage::CommonCommit,
        next.coordinate_count,
        MetalPersistentResidencySlidingDiagnosticPredicate::Commit, 1u,
        PersistentSlidingNoCoordinate);
    owner.quarantine_ticket = true;
    persistent_sliding_quarantine_ticket(next.ticket);
    return false;
  }
  if (next.coordinate_count >
      std::numeric_limits<std::uint64_t>::max() - owner.event_base) {
    record_first_failure(
        owner, MetalPersistentResidencySlidingDiagnosticStage::CommonCommit,
        next.coordinate_count,
        MetalPersistentResidencySlidingDiagnosticPredicate::Commit, 2u,
        PersistentSlidingNoCoordinate);
    return false;
  }
  owner.event_base += next.coordinate_count;
  owner.owner_nonce = next.owner_nonce;
  owner.pending_request = {};
  owner.rollback_request = {};
  owner.rollback_command = nil;
  owner.native_pending = false;
  owner.native_prepared = false;
  return true;
}

void abort_rearm(Owner &owner) noexcept {
  if (owner.native_prepared) {
    owner.command = owner.rollback_command;
    owner.prepared = owner.rollback_request;
    refresh_coordinates(owner, owner.prepared);
  }
  if (owner.native_pending && owner.pending_request.ticket != nullptr) {
    persistent_sliding_abort_ticket(owner.pending_request.ticket);
  }
  owner.pending_request = {};
  owner.native_pending = false;
  owner.rollback_request = {};
  owner.rollback_command = nil;
  owner.native_prepared = false;
}

#endif

} // namespace rund::node::accel::detail::metal_persistent_sliding

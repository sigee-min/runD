#include <rund/compute/pipeline.hpp>
#include <rund/counter.hpp>

#include "../local.hpp"
#include "internal.hpp"

#include <cstddef>
#include <memory>
#include <mutex>
#include <new>

namespace rund::compute::detail {

Result<std::shared_ptr<StateSnapshotState>>
snapshot_pipeline_state(const std::shared_ptr<PipelineState> &state) noexcept {
  if (!valid_pipeline(state)) {
    return Result<std::shared_ptr<StateSnapshotState>>::fail(
        Reason::PipelineInvalid);
  }
  std::lock_guard pipeline_lock{state->gate};
  if (state->samples != PipelineState::SampleState::Inactive) {
    return Result<std::shared_ptr<StateSnapshotState>>::fail(
        Reason::ProfileBusy);
  }
  if (state->publication == nullptr) {
    return Result<std::shared_ptr<StateSnapshotState>>::fail(
        Reason::PipelineInvalid);
  }
  std::lock_guard publication_lock{state->publication->gate};
  PipelinePublicationState &publication = *state->publication;
  if (publication.device_lost) {
    return Result<std::shared_ptr<StateSnapshotState>>::fail(
        Reason::DeviceLost);
  }
  if (!state->transactional || publication.state_pairs.empty()) {
    return Result<std::shared_ptr<StateSnapshotState>>::fail(
        Reason::PipelineInvalid);
  }
  if (state->phase == PipelinePhase::Running || publication.attempt_active) {
    return Result<std::shared_ptr<StateSnapshotState>>::fail(
        Reason::PipelineBusy);
  }
  try {
    auto snapshot = std::make_shared<StateSnapshotState>();
    snapshot->fields.reserve(publication.state_pairs.size());
    std::size_t bytes = 0u;
    const Status shaped = snapshot_detail::shape(publication, bytes);
    if (!shaped) {
      return Result<std::shared_ptr<StateSnapshotState>>::fail(shaped.reason());
    }
    const Status prepared = snapshot_detail::prepare_metadata(
        publication, publication.generation, bytes, publication.state_pairs.size(),
        *snapshot);
    if (!prepared) {
      return Result<std::shared_ptr<StateSnapshotState>>::fail(
          prepared.reason());
    }
    if (bytes != 0u) {
      // Every byte is filled by the following CPU copy or backend download.
      // Default-initialized std::byte[] avoids vector::resize's redundant
      // value-initialization pass without exposing the storage publicly.
      snapshot->bytes.reset(new std::byte[bytes]);
    }
    std::uint64_t transfer_count = 0u;
    const Status captured = snapshot_detail::capture_payload(
        *state, publication, *snapshot, transfer_count);
    (void)transfer_count;
    if (!captured) {
      return Result<std::shared_ptr<StateSnapshotState>>::fail(
          captured.reason());
    }
    state->stats.publication.snapshot_byte_count =
        ::rund::detail::counter::SaturatingAdd(
            state->stats.publication.snapshot_byte_count, bytes);
    state->stats.publication.snapshot_hash = snapshot->hash;
    return Result<std::shared_ptr<StateSnapshotState>>::success(
        std::move(snapshot));
  } catch (const std::bad_alloc &) {
    return Result<std::shared_ptr<StateSnapshotState>>::fail(
        Reason::BufferCapacity);
  }
}

Status snapshot_pipeline_into(
    const std::shared_ptr<PipelineState> &state,
    const std::shared_ptr<SnapshotStorageState> &storage) noexcept {
  if (!valid_pipeline(state) || storage == nullptr) {
    return Status::fail(Reason::PipelineInvalid);
  }
  std::unique_lock pipeline_lock{state->gate, std::try_to_lock};
  if (!pipeline_lock.owns_lock()) {
    return Status::fail(Reason::PipelineBusy);
  }
  if (state->samples != PipelineState::SampleState::Inactive) {
    return Status::fail(Reason::ProfileBusy);
  }
  if (state->publication == nullptr) {
    return Status::fail(Reason::PipelineInvalid);
  }
  std::unique_lock publication_lock{state->publication->gate, std::try_to_lock};
  if (!publication_lock.owns_lock()) {
    return Status::fail(Reason::PipelineBusy);
  }
  PipelinePublicationState &publication = *state->publication;
  if (publication.device_lost) {
    return Status::fail(Reason::DeviceLost);
  }
  if (!state->transactional || publication.state_pairs.empty()) {
    return Status::fail(Reason::PipelineInvalid);
  }
  if (state->phase == PipelinePhase::Running || publication.attempt_active) {
    return Status::fail(Reason::PipelineBusy);
  }
  std::unique_lock storage_lock{storage->gate, std::try_to_lock};
  if (!storage_lock.owns_lock()) {
    return Status::fail(Reason::PipelineBusy);
  }
  const std::uint8_t inactive = storage->active ^ std::uint8_t{1u};
  StateSnapshotState &snapshot = storage->banks[inactive];
  try {
    const Status prepared = snapshot_detail::prepare_metadata(
        publication, publication.generation, storage->byte_capacity,
        storage->field_capacity, snapshot);
    if (!prepared) {
      return prepared;
    }
    std::uint64_t transfer_count = 0u;
    const Status captured = snapshot_detail::capture_payload(
        *state, publication, snapshot, transfer_count);
    if (!captured) {
      return captured;
    }
    storage->active = inactive;
    storage->valid = true;
    ::rund::detail::counter::Accumulate(
        state->checkpoint_stats.reusable_snapshot_count, 1u);
    state->checkpoint_stats.reusable_snapshot_byte_count =
        ::rund::detail::counter::SaturatingAdd(
            state->checkpoint_stats.reusable_snapshot_byte_count,
            snapshot.byte_count);
    state->checkpoint_stats.reusable_snapshot_hash = snapshot.hash;
    state->checkpoint_stats.reusable_snapshot_transfer_count =
        ::rund::detail::counter::SaturatingAdd(
            state->checkpoint_stats.reusable_snapshot_transfer_count,
            transfer_count);
    return Status::success();
  } catch (const std::bad_alloc &) {
    // Metadata capacity was reserved at storage creation. This is defensive;
    // publication still remains on the previously valid bank.
    return Status::fail(Reason::BufferCapacity);
  }
}

Status restore_pipeline_state(
    const std::shared_ptr<PipelineState> &state,
    const std::shared_ptr<StateSnapshotState> &snapshot) noexcept {
  if (!valid_pipeline(state) || snapshot == nullptr) {
    return Status::fail(Reason::PipelineInvalid);
  }
  std::lock_guard pipeline_lock{state->gate};
  if (state->samples != PipelineState::SampleState::Inactive) {
    return Status::fail(Reason::ProfileBusy);
  }
  if (state->publication == nullptr) {
    return Status::fail(Reason::PipelineInvalid);
  }
  std::lock_guard publication_lock{state->publication->gate};
  return snapshot_detail::restore_locked(*state, *state->publication,
                                         *snapshot);
}

Status restore_pipeline_state(
    const std::shared_ptr<PipelineState> &state,
    const std::shared_ptr<SnapshotStorageState> &storage) noexcept {
  if (!valid_pipeline(state) || storage == nullptr) {
    return Status::fail(Reason::PipelineInvalid);
  }
  std::unique_lock pipeline_lock{state->gate, std::try_to_lock};
  if (!pipeline_lock.owns_lock()) {
    return Status::fail(Reason::PipelineBusy);
  }
  if (state->samples != PipelineState::SampleState::Inactive) {
    return Status::fail(Reason::ProfileBusy);
  }
  if (state->publication == nullptr) {
    return Status::fail(Reason::PipelineInvalid);
  }
  std::unique_lock publication_lock{state->publication->gate, std::try_to_lock};
  if (!publication_lock.owns_lock()) {
    return Status::fail(Reason::PipelineBusy);
  }
  std::unique_lock storage_lock{storage->gate, std::try_to_lock};
  if (!storage_lock.owns_lock()) {
    return Status::fail(Reason::PipelineBusy);
  }
  if (!storage->valid || storage->active >= storage->banks.size()) {
    return Status::fail(Reason::PipelineInvalid);
  }
  return snapshot_detail::restore_locked(
      *state, *state->publication, storage->banks[storage->active]);
}

bool snapshot_valid(
    const std::shared_ptr<StateSnapshotState> &snapshot) noexcept {
  // StateSnapshotState has no public construction or mutation path. Its
  // payload is sealed before publication, while restore() performs the full
  // schema/hash validation at the trust boundary. Keep ordinary immutable
  // accessors O(1) instead of re-hashing every saved byte on each query.
  return snapshot != nullptr;
}

std::uint64_t snapshot_generation(
    const std::shared_ptr<StateSnapshotState> &snapshot) noexcept {
  return snapshot == nullptr ? 0u : snapshot->generation;
}

graph::Fingerprint snapshot_fingerprint(
    const std::shared_ptr<StateSnapshotState> &snapshot) noexcept {
  return snapshot == nullptr ? graph::Fingerprint{} : snapshot->fingerprint;
}

std::uint64_t
snapshot_hash(const std::shared_ptr<StateSnapshotState> &snapshot) noexcept {
  return snapshot == nullptr ? 0u : snapshot->hash;
}

} // namespace rund::compute::detail

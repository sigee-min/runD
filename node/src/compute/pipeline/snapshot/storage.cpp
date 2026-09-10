#include <rund/compute/pipeline.hpp>

#include "internal.hpp"

#include "../local.hpp"

#include <cstddef>
#include <memory>
#include <mutex>
#include <new>

namespace rund::compute::detail {

Result<std::shared_ptr<SnapshotStorageState>>
make_snapshot_storage(const std::shared_ptr<PipelineState> &state,
                      const std::size_t byte_capacity,
                      const bool exact) noexcept {
  if (!valid_pipeline(state)) {
    return Result<std::shared_ptr<SnapshotStorageState>>::fail(
        Reason::PipelineInvalid);
  }
  std::lock_guard pipeline_lock{state->gate};
  if (state->samples != PipelineState::SampleState::Inactive) {
    return Result<std::shared_ptr<SnapshotStorageState>>::fail(
        Reason::ProfileBusy);
  }
  if (state->publication == nullptr) {
    return Result<std::shared_ptr<SnapshotStorageState>>::fail(
        Reason::PipelineInvalid);
  }
  std::lock_guard publication_lock{state->publication->gate};
  const PipelinePublicationState &publication = *state->publication;
  if (publication.device_lost) {
    return Result<std::shared_ptr<SnapshotStorageState>>::fail(
        Reason::DeviceLost);
  }
  if (!state->transactional || publication.state_pairs.empty()) {
    return Result<std::shared_ptr<SnapshotStorageState>>::fail(
        Reason::PipelineInvalid);
  }
  if (state->phase == PipelinePhase::Running || publication.attempt_active) {
    return Result<std::shared_ptr<SnapshotStorageState>>::fail(
        Reason::PipelineBusy);
  }
  std::size_t required_bytes = 0u;
  const Status shaped = snapshot_detail::shape(publication, required_bytes);
  if (!shaped) {
    return Result<std::shared_ptr<SnapshotStorageState>>::fail(shaped.reason());
  }
  const std::size_t capacity = exact ? required_bytes : byte_capacity;
  try {
    auto storage = std::make_shared<SnapshotStorageState>();
    storage->byte_capacity = capacity;
    storage->field_capacity = publication.state_pairs.size();
    for (StateSnapshotState &bank : storage->banks) {
      bank.fields.reserve(storage->field_capacity);
      if (capacity != 0u) {
        bank.bytes.reset(new std::byte[capacity]);
      }
    }
    return Result<std::shared_ptr<SnapshotStorageState>>::success(
        std::move(storage));
  } catch (const std::bad_alloc &) {
    return Result<std::shared_ptr<SnapshotStorageState>>::fail(
        Reason::BufferCapacity);
  }
}

bool snapshot_storage_valid(
    const std::shared_ptr<SnapshotStorageState> &storage) noexcept {
  return storage != nullptr;
}

bool snapshot_storage_has_snapshot(
    const std::shared_ptr<SnapshotStorageState> &storage) noexcept {
  if (storage == nullptr) {
    return false;
  }
  std::lock_guard lock{storage->gate};
  return storage->valid && storage->active < storage->banks.size();
}

std::uint64_t snapshot_storage_generation(
    const std::shared_ptr<SnapshotStorageState> &storage) noexcept {
  if (storage == nullptr) {
    return 0u;
  }
  std::lock_guard lock{storage->gate};
  return !storage->valid || storage->active >= storage->banks.size()
             ? 0u
             : storage->banks[storage->active].generation;
}

graph::Fingerprint snapshot_storage_fingerprint(
    const std::shared_ptr<SnapshotStorageState> &storage) noexcept {
  if (storage == nullptr) {
    return {};
  }
  std::lock_guard lock{storage->gate};
  return !storage->valid || storage->active >= storage->banks.size()
             ? graph::Fingerprint{}
             : storage->banks[storage->active].fingerprint;
}

std::uint64_t snapshot_storage_hash(
    const std::shared_ptr<SnapshotStorageState> &storage) noexcept {
  if (storage == nullptr) {
    return 0u;
  }
  std::lock_guard lock{storage->gate};
  return !storage->valid || storage->active >= storage->banks.size()
             ? 0u
             : storage->banks[storage->active].hash;
}

std::size_t snapshot_storage_capacity(
    const std::shared_ptr<SnapshotStorageState> &storage) noexcept {
  return storage == nullptr ? 0u : storage->byte_capacity;
}

std::size_t snapshot_storage_field_capacity(
    const std::shared_ptr<SnapshotStorageState> &storage) noexcept {
  return storage == nullptr ? 0u : storage->field_capacity;
}

} // namespace rund::compute::detail

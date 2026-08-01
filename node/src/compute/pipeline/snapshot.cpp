#include <rund/compute/pipeline.hpp>
#include <rund/counter.hpp>

#include "../../hash/fnv.hpp"
#include "../backend.hpp"
#include "../size.hpp"
#include "../status.hpp"
#include "../type.hpp"
#include "claim.hpp"
#include "local.hpp"
#include "state.hpp"

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <functional>
#include <limits>
#include <memory>
#include <mutex>
#include <new>
#include <span>

namespace rund::compute::detail {
namespace {

void mix(std::uint64_t &hash, const std::uint64_t value) noexcept {
  ::rund::node::hash_detail::MixU64(hash, value);
}

[[nodiscard]] std::uint64_t
snapshot_hash(const StateSnapshotState &snapshot) noexcept {
  std::uint64_t hash = ::rund::node::hash_detail::kFnvOffset;
  mix(hash, snapshot.fingerprint.hi);
  mix(hash, snapshot.fingerprint.lo);
  mix(hash, snapshot.generation);
  mix(hash, snapshot.fields.size());
  for (const PipelineSnapshotField &field : snapshot.fields) {
    mix(hash, static_cast<std::uint64_t>(field.type));
    mix(hash, field.format.integer_bits);
    mix(hash, field.format.fraction_bits);
    mix(hash, static_cast<std::uint64_t>(field.format.rounding));
    mix(hash, static_cast<std::uint64_t>(field.format.overflow));
    mix(hash, static_cast<std::uint64_t>(field.format.approximation));
    mix(hash, field.count);
    mix(hash, field.offset);
    mix(hash, field.bytes);
    mix(hash, field.payload_hash);
  }
  return hash;
}

[[nodiscard]] bool
valid_snapshot_layout(const StateSnapshotState &snapshot) noexcept {
  if (!snapshot.fingerprint || snapshot.fields.empty() ||
      (snapshot.byte_count != 0u && snapshot.bytes == nullptr)) {
    return false;
  }
  std::size_t offset = 0u;
  for (const PipelineSnapshotField &field : snapshot.fields) {
    const std::size_t element_bytes = type_bytes(field.type);
    std::size_t field_bytes = 0u;
    if (element_bytes == 0u || field.offset != offset ||
        !size::multiply(field.count, element_bytes, field_bytes) ||
        field.bytes != field_bytes ||
        field.bytes >
            snapshot.byte_count - std::min(snapshot.byte_count, offset) ||
        field.payload_hash !=
            (field.bytes == 0u
                 ? ::rund::node::hash_detail::kFnvOffset
                 : ::rund::node::hash_detail::HashBytes(
                       snapshot.bytes.get() + field.offset, field.bytes))) {
      return false;
    }
    if (!size::add(offset, field.bytes, offset)) {
      return false;
    }
  }
  return offset == snapshot.byte_count &&
         snapshot.hash == snapshot_hash(snapshot);
}

[[nodiscard]] BufferState *published_buffer(PipelinePublicationState &state,
                                            const std::size_t index) noexcept {
  if (index >= state.state_pairs.size()) {
    return nullptr;
  }
  PipelineStatePair &pair = state.state_pairs[index];
  return (state.parity == 0u ? pair.first : pair.second).get();
}

template <typename Transfer>
void record_staging(PipelineState &state, const Transfer &transfer) noexcept {
  state.read_staging_bytes = ::rund::detail::counter::SaturatingAdd(
      state.read_staging_bytes, transfer.staging_bytes);
  state.read_staging_reused = ::rund::detail::counter::SaturatingAdd(
      state.read_staging_reused, transfer.staging_reused_bytes);
  state.read_staging_peak =
      std::max(state.read_staging_peak, transfer.staging_peak_bytes);
  state.read_staging_budget =
      std::max(state.read_staging_budget, transfer.staging_budget);
  ::rund::detail::counter::Accumulate(state.stats.buffer_allocations,
                                      transfer.buffer_allocations);
  ::rund::detail::counter::Accumulate(state.stats.buffer_reuses,
                                      transfer.buffer_reuses);
  ::rund::detail::counter::Accumulate(state.stats.command_submits,
                                      transfer.command_submits);
}

void record_download(PipelineState &state, const std::size_t bytes,
                     const DownloadResult &transfer,
                     const std::size_t events = 1u) noexcept {
  record_staging(state, transfer);
  ::rund::detail::counter::Accumulate(state.stats.readback_ns,
                                      transfer.readback_ns);
  ::rund::detail::counter::Accumulate(state.stats.download_events,
                                      bytes == 0u ? 0u : events);
  ::rund::detail::counter::Accumulate(state.stats.downloaded_bytes, bytes);
  state.read_transfer_bytes =
      ::rund::detail::counter::SaturatingAdd(state.read_transfer_bytes, bytes);
  state.read_transfer_peak =
      std::max(state.read_transfer_peak, static_cast<std::uint64_t>(bytes));
  record_transfer(*state.device, bytes);
}

void record_upload(PipelineState &state, const std::size_t bytes,
                   const UploadResult &transfer) noexcept {
  record_staging(state, transfer);
  ::rund::detail::counter::Accumulate(state.stats.uploaded_bytes, bytes);
  state.read_transfer_bytes =
      ::rund::detail::counter::SaturatingAdd(state.read_transfer_bytes, bytes);
  state.read_transfer_peak =
      std::max(state.read_transfer_peak, static_cast<std::uint64_t>(bytes));
  record_transfer(*state.device, bytes);
}

[[nodiscard]] Status snapshot_shape(const PipelinePublicationState &publication,
                                    std::size_t &bytes) noexcept {
  if (publication.state_pairs.empty()) {
    return Status::fail(Reason::PipelineInvalid);
  }
  if (publication.state_pairs.size() > PipelineLeafCapacity) {
    return Status::fail(Reason::PipelineCapacity);
  }
  bytes = 0u;
  for (const PipelineStatePair &pair : publication.state_pairs) {
    if (!size::add(bytes, pair.bytes, bytes)) {
      return Status::fail(Reason::BufferCapacity);
    }
  }
  return Status::success();
}

[[nodiscard]] Status prepare_snapshot_metadata(
    const PipelinePublicationState &publication, const std::uint64_t generation,
    const std::size_t byte_capacity, const std::size_t field_capacity,
    StateSnapshotState &snapshot) {
  std::size_t bytes = 0u;
  const Status shaped = snapshot_shape(publication, bytes);
  if (!shaped) {
    return shaped;
  }
  if (bytes > byte_capacity ||
      publication.state_pairs.size() > field_capacity ||
      snapshot.fields.capacity() < publication.state_pairs.size()) {
    return Status::fail(Reason::BufferCapacity);
  }
  snapshot.fields.clear();
  snapshot.fingerprint = publication.fingerprint;
  snapshot.generation = generation;
  snapshot.byte_count = bytes;
  snapshot.hash = 0u;
  std::size_t offset = 0u;
  for (const PipelineStatePair &pair : publication.state_pairs) {
    snapshot.fields.push_back(PipelineSnapshotField{
        .type = pair.type,
        .format = pair.format,
        .count = pair.count,
        .offset = offset,
        .bytes = pair.bytes,
        .payload_hash = ::rund::node::hash_detail::kFnvOffset,
    });
    offset += pair.bytes;
  }
  return Status::success();
}

[[nodiscard]] Status capture_snapshot_payload(
    PipelineState &state, PipelinePublicationState &publication,
    StateSnapshotState &snapshot, std::uint64_t &transfer_count) noexcept {
  if (snapshot.fields.size() != publication.state_pairs.size() ||
      (snapshot.byte_count != 0u && snapshot.bytes == nullptr)) {
    return Status::fail(Reason::PipelineInvalid);
  }
  std::array<BufferClaim, PipelineLeafCapacity> claim_storage{};
  for (std::size_t index = 0u; index < publication.state_pairs.size();
       ++index) {
    claim_storage[index] = BufferClaim{
        .buffer = published_buffer(publication, index), .write = false};
  }
  const std::span<const BufferClaim> claims{claim_storage.data(),
                                            publication.state_pairs.size()};
  const Status claimed = acquire_claims(*state.device, claims);
  if (!claimed) {
    return claimed;
  }
  ClaimGuard guard{*state.device, claims};
  std::array<DownloadRequest, PipelineLeafCapacity> download_storage{};
  std::size_t download_count = 0u;
  for (std::size_t index = 0u; index < claims.size(); ++index) {
    BufferState *const source = claims[index].buffer;
    PipelineSnapshotField &field = snapshot.fields[index];
    if (field.bytes == 0u) {
      continue;
    }
    void *const destination = snapshot.bytes.get() + field.offset;
    if (state.device->backend == Backend::Cpu) {
      const CpuBufferState *const cpu =
          source == nullptr ? nullptr : cpu_buffer(*source);
      if (cpu == nullptr || cpu->data == nullptr || cpu->bytes < field.bytes) {
        return Status::fail(Reason::TransferInvalid);
      }
      field.payload_hash = ::rund::node::hash_detail::CopyHash(
          cpu->data.get(), destination, field.bytes);
    } else {
      if (source == nullptr) {
        return Status::fail(Reason::TransferInvalid);
      }
      download_storage[download_count++] = DownloadRequest{
          .buffer = source,
          .data = destination,
          .bytes = field.bytes,
          .payload_hash = &field.payload_hash,
      };
    }
  }
  const std::span<const DownloadRequest> downloads{download_storage.data(),
                                                   download_count};
  if (state.device->backend != Backend::Cpu && !downloads.empty()) {
    if (state.device->ops == nullptr ||
        state.device->ops->download_batch == nullptr) {
      return Status::fail(Reason::TransferInvalid);
    }
    const DownloadResult transfer =
        state.device->ops->download_batch(*state.device, downloads);
    if (!transfer.status) {
      if (transfer.status.reason() == Reason::DeviceLost) {
        publication.device_lost = true;
      }
      return transfer.status;
    }
    if (!transfer.payload_hash_valid) {
      return Status::fail(Reason::TransferInvalid);
    }
    record_download(state, snapshot.byte_count, transfer, downloads.size());
    transfer_count = std::max<std::uint64_t>(1u, transfer.command_submits);
  }
  snapshot.hash = snapshot_hash(snapshot);
  return Status::success();
}

[[nodiscard]] Status
restore_host_snapshot_locked(PipelineState &state,
                             PipelinePublicationState &publication,
                             const StateSnapshotState &snapshot) noexcept {
  if (publication.device_lost) {
    return Status::fail(Reason::DeviceLost);
  }
  if (publication.attempt_active || state.phase == PipelinePhase::Running) {
    return Status::fail(Reason::PipelineBusy);
  }
  if (!state.transactional || publication.state_pairs.empty() ||
      state.phase != PipelinePhase::Ready) {
    return Status::fail(state.phase == PipelinePhase::Poisoned
                            ? Reason::PipelinePoisoned
                            : Reason::PipelineInvalid);
  }
  if (publication.payload_epoch == std::numeric_limits<std::uint64_t>::max()) {
    return Status::fail(Reason::PipelineCapacity);
  }
  if (snapshot.generation > PipelineGenerationCapacity) {
    return Status::fail(Reason::PipelineCapacity);
  }
  if (!valid_snapshot_layout(snapshot) ||
      snapshot.fingerprint != publication.fingerprint ||
      snapshot.fields.size() != publication.state_pairs.size()) {
    return Status::fail(Reason::PipelineInvalid);
  }
  if (publication.state_pairs.size() > PipelineLeafCapacity) {
    return Status::fail(Reason::PipelineCapacity);
  }
  std::array<BufferClaim, PipelineLeafCapacity * 2u> claim_storage{};
  for (std::size_t index = 0u; index < publication.state_pairs.size();
       ++index) {
    const PipelineStatePair &pair = publication.state_pairs[index];
    const PipelineSnapshotField &field = snapshot.fields[index];
    if (pair.type != field.type || pair.format != field.format ||
        pair.count != field.count || pair.bytes != field.bytes) {
      return Status::fail(Reason::PipelineInvalid);
    }
    claim_storage[index * 2u] =
        BufferClaim{.buffer = pair.first.get(), .write = true};
    claim_storage[index * 2u + 1u] =
        BufferClaim{.buffer = pair.second.get(), .write = true};
  }
  const std::span<const BufferClaim> claims{
      claim_storage.data(), publication.state_pairs.size() * 2u};
  const Status claimed = acquire_claims(*state.device, claims, false);
  if (!claimed) {
    return claimed;
  }
  ClaimGuard guard{*state.device, claims};
  Status restored = Status::success();
  std::size_t copied = 0u;
  if (state.device->backend == Backend::Cpu) {
    for (std::size_t field_index = 0u;
         restored && field_index < snapshot.fields.size(); ++field_index) {
      const PipelineSnapshotField &field = snapshot.fields[field_index];
      for (std::size_t copy = 0u; copy < 2u; ++copy) {
        BufferState *const destination = claims[field_index * 2u + copy].buffer;
        if (field.bytes == 0u) {
          continue;
        }
        const void *const source = snapshot.bytes.get() + field.offset;
        CpuBufferState *const cpu =
            destination == nullptr ? nullptr : cpu_buffer(*destination);
        if (cpu == nullptr || cpu->data == nullptr ||
            cpu->bytes < field.bytes) {
          restored = Status::fail(Reason::TransferInvalid);
          break;
        }
        std::memcpy(cpu->data.get(), source, field.bytes);
        copied = ::rund::detail::counter::SaturatingAdd(copied, field.bytes);
        record_transfer(*state.device, field.bytes);
      }
    }
  } else {
    std::array<UploadRequest, PipelineLeafCapacity * 2u> upload_storage{};
    std::size_t upload_count = 0u;
    for (std::size_t field_index = 0u; field_index < snapshot.fields.size();
         ++field_index) {
      const PipelineSnapshotField &field = snapshot.fields[field_index];
      const void *const source =
          field.bytes == 0u ? nullptr : snapshot.bytes.get() + field.offset;
      for (std::size_t copy = 0u; copy < 2u; ++copy) {
        BufferState *const destination = claims[field_index * 2u + copy].buffer;
        if (destination == nullptr) {
          restored = Status::fail(Reason::TransferInvalid);
          break;
        }
        if (field.bytes != 0u) {
          upload_storage[upload_count++] = UploadRequest{
              .buffer = destination, .data = source, .bytes = field.bytes};
        }
      }
      if (!restored) {
        break;
      }
    }
    const std::span<const UploadRequest> uploads{upload_storage.data(),
                                                 upload_count};
    if (restored && !uploads.empty() &&
        (state.device->ops == nullptr ||
         state.device->ops->upload_batch == nullptr)) {
      restored = Status::fail(Reason::TransferInvalid);
    }
    if (restored && !uploads.empty()) {
      const UploadResult transfer = state.device->ops->upload_batch(
          *state.device, uploads,
          node::accel::detail::TransferCompletion::Complete);
      restored = transfer.status;
      if (restored) {
        for (const UploadRequest upload : uploads) {
          copied = ::rund::detail::counter::SaturatingAdd(copied, upload.bytes);
        }
        record_upload(state, copied, transfer);
      }
    }
  }
  if (restored) {
    restored = seed_pipeline_generations(state, snapshot.generation, 0u);
  }
  if (!restored) {
    if (restored.reason() == Reason::DeviceLost) {
      publication.device_lost = true;
    }
    publish_claims(*state.device, claims, false, true);
    guard.dismiss();
    state.failure = restored.reason();
    state.phase = PipelinePhase::Poisoned;
    return restored;
  }
  {
    std::lock_guard claim_lock{state.device->claims->gate};
    for (const BufferClaim claim : claims) {
      claim.buffer->poisoned = false;
      ++claim.buffer->generation;
      claim.buffer->writer = false;
    }
  }
  guard.dismiss();
  publication.parity = 0u;
  publication.generation = snapshot.generation;
  ++publication.payload_epoch;
  publication.device_lost = false;
  close_pipeline_observation_epoch(state);
  state.failure = Reason::Ok;
  state.phase = PipelinePhase::Ready;
  state.control_poisoned = false;
  state.stats.publication.generation = publication.generation;
  state.stats.publication.restore_byte_count =
      ::rund::detail::counter::SaturatingAdd(
          state.stats.publication.restore_byte_count, copied);
  return Status::success();
}

} // namespace

Result<std::shared_ptr<StateSnapshotState>>
snapshot_pipeline_state(const std::shared_ptr<PipelineState> &state) noexcept {
  if (!valid_pipeline(state)) {
    return Result<std::shared_ptr<StateSnapshotState>>::fail(
        Reason::PipelineInvalid);
  }
  std::lock_guard pipeline_lock{state->gate};
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
    const Status shaped = snapshot_shape(publication, bytes);
    if (!shaped) {
      return Result<std::shared_ptr<StateSnapshotState>>::fail(shaped.reason());
    }
    const Status prepared =
        prepare_snapshot_metadata(publication, publication.generation, bytes,
                                  publication.state_pairs.size(), *snapshot);
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
    const Status captured = capture_snapshot_payload(*state, publication,
                                                     *snapshot, transfer_count);
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

Result<std::shared_ptr<PipelinePublicationState>>
latest_pipeline_state(const std::shared_ptr<PipelineState> &state) noexcept {
  if (!valid_pipeline(state)) {
    return Result<std::shared_ptr<PipelinePublicationState>>::fail(
        Reason::PipelineInvalid);
  }
  std::lock_guard pipeline_lock{state->gate};
  if (state->publication == nullptr) {
    return Result<std::shared_ptr<PipelinePublicationState>>::fail(
        Reason::PipelineInvalid);
  }
  std::lock_guard publication_lock{state->publication->gate};
  if (state->publication->device_lost) {
    return Result<std::shared_ptr<PipelinePublicationState>>::fail(
        Reason::DeviceLost);
  }
  if (!state->transactional || state->publication->state_pairs.empty() ||
      state->publication->device == nullptr ||
      !state->publication->fingerprint) {
    return Result<std::shared_ptr<PipelinePublicationState>>::fail(
        Reason::PipelineInvalid);
  }
  ::rund::detail::counter::Accumulate(
      state->checkpoint_stats.device_state_acquire_count, 1u);
  return Result<std::shared_ptr<PipelinePublicationState>>::success(
      state->publication);
}

Result<std::shared_ptr<SnapshotStorageState>>
make_snapshot_storage(const std::shared_ptr<PipelineState> &state,
                      const std::size_t byte_capacity,
                      const bool exact) noexcept {
  if (!valid_pipeline(state)) {
    return Result<std::shared_ptr<SnapshotStorageState>>::fail(
        Reason::PipelineInvalid);
  }
  std::lock_guard pipeline_lock{state->gate};
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
  const Status shaped = snapshot_shape(publication, required_bytes);
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
    const Status prepared = prepare_snapshot_metadata(
        publication, publication.generation, storage->byte_capacity,
        storage->field_capacity, snapshot);
    if (!prepared) {
      return prepared;
    }
    std::uint64_t transfer_count = 0u;
    const Status captured =
        capture_snapshot_payload(*state, publication, snapshot, transfer_count);
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
  if (state->publication == nullptr) {
    return Status::fail(Reason::PipelineInvalid);
  }
  std::lock_guard publication_lock{state->publication->gate};
  return restore_host_snapshot_locked(*state, *state->publication, *snapshot);
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
  return restore_host_snapshot_locked(*state, *state->publication,
                                      storage->banks[storage->active]);
}

Status restore_pipeline_state(
    const std::shared_ptr<PipelineState> &state,
    const std::shared_ptr<PipelinePublicationState> &source) noexcept {
  if (!valid_pipeline(state) || source == nullptr) {
    return Status::fail(Reason::PipelineInvalid);
  }
  std::unique_lock pipeline_lock{state->gate, std::try_to_lock};
  if (!pipeline_lock.owns_lock()) {
    return Status::fail(Reason::PipelineBusy);
  }
  const std::shared_ptr<PipelinePublicationState> target = state->publication;
  if (target == nullptr) {
    return Status::fail(Reason::PipelineInvalid);
  }

  std::unique_lock<std::mutex> target_lock{target->gate, std::defer_lock};
  std::unique_lock<std::mutex> source_lock{source->gate, std::defer_lock};
  if (target == source) {
    if (!target_lock.try_lock()) {
      return Status::fail(Reason::PipelineBusy);
    }
  } else if (std::less<const PipelinePublicationState *>{}(target.get(),
                                                           source.get())) {
    if (!target_lock.try_lock() || !source_lock.try_lock()) {
      return Status::fail(Reason::PipelineBusy);
    }
  } else if (!source_lock.try_lock() || !target_lock.try_lock()) {
    return Status::fail(Reason::PipelineBusy);
  }

  if (target->device_lost || source->device_lost) {
    return Status::fail(Reason::DeviceLost);
  }
  if (target->attempt_active || source->attempt_active ||
      state->phase == PipelinePhase::Running) {
    return Status::fail(Reason::PipelineBusy);
  }
  if (!state->transactional || target->state_pairs.empty() ||
      source->state_pairs.empty() || state->phase != PipelinePhase::Ready) {
    return Status::fail(state->phase == PipelinePhase::Poisoned
                            ? Reason::PipelinePoisoned
                            : Reason::PipelineInvalid);
  }
  if (source->generation > PipelineGenerationCapacity) {
    return Status::fail(Reason::PipelineCapacity);
  }
  if (target->device == nullptr || source->device == nullptr ||
      target->device != state->device || source->device != state->device) {
    return Status::fail(Reason::BindingDeviceMismatch);
  }
  if (target->fingerprint != source->fingerprint ||
      target->state_pairs.size() != source->state_pairs.size() ||
      target->state_pairs.size() > PipelineLeafCapacity) {
    return Status::fail(Reason::PipelineInvalid);
  }

  bool ordered_owners_equal = true;
  for (std::size_t index = 0u; index < target->state_pairs.size(); ++index) {
    const PipelineStatePair &destination = target->state_pairs[index];
    const PipelineStatePair &published = source->state_pairs[index];
    if (destination.type != published.type ||
        destination.format != published.format ||
        destination.count != published.count ||
        destination.bytes != published.bytes) {
      return Status::fail(Reason::PipelineInvalid);
    }
    ordered_owners_equal = ordered_owners_equal &&
                           destination.first == published.first &&
                           destination.second == published.second;
  }

  if (ordered_owners_equal) {
    if (target != source && !state->preparing) {
      // Replacing a live authority would strand any handle already issued by
      // the destination. Nothing has been mutated yet, so reject atomically
      // and preserve the destination's current publication and phase.
      return Status::fail(Reason::PipelineInvalid);
    }
    const Status seeded =
        seed_pipeline_generations(*state, source->generation, source->parity);
    if (!seeded) {
      if (seeded.reason() == Reason::DeviceLost) {
        target->device_lost = true;
        source->device_lost = true;
      }
      state->failure = seeded.reason();
      state->phase = PipelinePhase::Poisoned;
      return seeded;
    }
    if (target != source) {
      state->publication = source;
    }
    state->failure = Reason::Ok;
    state->control_poisoned = false;
    state->stats.publication.generation = source->generation;
    ::rund::detail::counter::Accumulate(
        state->checkpoint_stats.device_state_rebase_count, 1u);
    return Status::success();
  }

  if (target->payload_epoch == std::numeric_limits<std::uint64_t>::max()) {
    return Status::fail(Reason::PipelineCapacity);
  }

  // The copy path is deliberately all-disjoint. Any partial alias (including
  // reversed pair orientation) would require a cycle-breaking temporary and
  // would leave two independent selectors over shared owners.
  for (const PipelineStatePair &destination : target->state_pairs) {
    for (const PipelineStatePair &published : source->state_pairs) {
      if (destination.first == published.first ||
          destination.first == published.second ||
          destination.second == published.first ||
          destination.second == published.second) {
        return Status::fail(Reason::BindingDuplicate);
      }
    }
  }

  const std::size_t pair_count = target->state_pairs.size();
  std::array<BufferClaim, PipelineLeafCapacity * 2u> claim_storage{};
  std::array<CopyRequest, PipelineLeafCapacity> copy_storage{};
  std::size_t copy_count = 0u;
  std::size_t copied = 0u;
  for (std::size_t index = 0u; index < pair_count; ++index) {
    const PipelineStatePair &published = source->state_pairs[index];
    const PipelineStatePair &destination = target->state_pairs[index];
    BufferState *const source_buffer =
        (source->parity == 0u ? published.first : published.second).get();
    BufferState *const target_buffer = destination.first.get();
    claim_storage[index] = BufferClaim{.buffer = source_buffer, .write = false};
    claim_storage[pair_count + index] =
        BufferClaim{.buffer = target_buffer, .write = true};
    if (published.bytes != 0u) {
      copy_storage[copy_count++] = CopyRequest{
          .source = source_buffer,
          .target = target_buffer,
          .bytes = published.bytes,
      };
      copied = ::rund::detail::counter::SaturatingAdd(copied, published.bytes);
    }
  }
  const std::span<const BufferClaim> claims{claim_storage.data(),
                                            pair_count * 2u};
  const Status claimed = acquire_claims(*state->device, claims);
  if (!claimed) {
    return claimed;
  }
  ClaimGuard guard{*state->device, claims};
  Status restored = Status::success();
  std::uint64_t copy_commands = 0u;
  if (state->device->backend == Backend::Cpu) {
    for (std::size_t index = 0u; index < copy_count; ++index) {
      const CopyRequest request = copy_storage[index];
      const CpuBufferState *const source_cpu = cpu_buffer(*request.source);
      CpuBufferState *const target_cpu = cpu_buffer(*request.target);
      if (source_cpu == nullptr || target_cpu == nullptr ||
          source_cpu->data == nullptr || target_cpu->data == nullptr ||
          source_cpu->bytes < request.bytes ||
          target_cpu->bytes < request.bytes) {
        restored = Status::fail(Reason::TransferInvalid);
        break;
      }
      std::memcpy(target_cpu->data.get(), source_cpu->data.get(),
                  request.bytes);
    }
  } else if (copy_count != 0u) {
    if (state->device->ops == nullptr ||
        state->device->ops->copy_batch == nullptr) {
      restored = Status::fail(Reason::TransferInvalid);
    } else {
      const CopyResult result = state->device->ops->copy_batch(
          *state->device,
          std::span<const CopyRequest>{copy_storage.data(), copy_count});
      restored = result.status;
      copy_commands = result.command_submits;
      ::rund::detail::counter::Accumulate(state->stats.command_submits,
                                          result.command_submits);
    }
  }
  if (restored) {
    restored = seed_pipeline_generations(*state, source->generation, 0u);
  }
  if (!restored) {
    if (restored.reason() == Reason::DeviceLost) {
      target->device_lost = true;
      source->device_lost = true;
    }
    publish_claims(*state->device, claims, false, true);
    guard.dismiss();
    state->failure = restored.reason();
    state->phase = PipelinePhase::Poisoned;
    return restored;
  }
  {
    std::lock_guard claim_lock{state->device->claims->gate};
    for (const BufferClaim claim : claims) {
      if (claim.write) {
        claim.buffer->poisoned = false;
        ++claim.buffer->generation;
        claim.buffer->writer = false;
      } else if (claim.buffer->readers != 0u) {
        --claim.buffer->readers;
      }
    }
  }
  guard.dismiss();
  target->parity = 0u;
  target->generation = source->generation;
  ++target->payload_epoch;
  target->device_lost = false;
  close_pipeline_observation_epoch(*state);
  state->failure = Reason::Ok;
  state->control_poisoned = false;
  state->stats.publication.generation = target->generation;
  state->checkpoint_stats.device_state_copy_byte_count =
      ::rund::detail::counter::SaturatingAdd(
          state->checkpoint_stats.device_state_copy_byte_count, copied);
  state->checkpoint_stats.device_state_copy_command_count =
      ::rund::detail::counter::SaturatingAdd(
          state->checkpoint_stats.device_state_copy_command_count,
          copy_commands);
  return Status::success();
}

Status seed_pipeline_generations(PipelineState &state,
                                 const std::uint64_t generation,
                                 const std::uint8_t parity) noexcept {
  if (generation > PipelineGenerationCapacity) {
    return Status::fail(Reason::PipelineCapacity);
  }
  if (parity > 1u) {
    return Status::fail(Reason::PipelineInvalid);
  }
  // Empty and CPU Pipelines intentionally own no native generation control.
  if (state.device->backend == Backend::Cpu || state.active_step_count == 0u) {
    state.native_generation = generation;
    state.native_parity = parity;
    return Status::success();
  }
  const DeviceOps *const ops = state.device->ops;
  if (ops == nullptr || ops->seed_pipeline_generation == nullptr) {
    return Status::fail(Reason::PipelineInvalid);
  }
  const std::uint32_t current = static_cast<std::uint32_t>(generation);
  const std::uint32_t selected_seed =
      state.transactional ? current - std::uint32_t{1u} : current;
  const std::uint32_t other_seed = current;
  const std::uint32_t primary_seed =
      state.transactional && parity != 0u ? other_seed : selected_seed;
  const std::uint32_t alternate_seed =
      state.transactional && parity != 0u ? selected_seed : other_seed;
  const rund::AccelCheck primary =
      ops->seed_pipeline_generation(state.prepared, primary_seed);
  const rund::AccelCheck alternate =
      primary.ok && state.transactional
          ? ops->seed_pipeline_generation(state.alternate_prepared,
                                          alternate_seed)
          : primary;
  if (!alternate.ok) {
    return Status::fail(
        project_reason(alternate.reason, Reason::PipelineInvalid));
  }
  state.native_generation = generation;
  state.native_parity = parity;
  return Status::success();
}

bool rebase_failed_pipeline_generation(PipelineState &state,
                                       const bool submitted,
                                       const Reason failure) noexcept {
  if (!submitted || state.device->backend == Backend::Cpu ||
      state.active_step_count == 0u || failure == Reason::DeviceLost) {
    return true;
  }
  const DeviceOps *const ops = state.device->ops;
  const node::accel::detail::PreparedKernelPipeline &selected =
      state.transactional && state.attempt_parity != 0u
          ? state.alternate_prepared
          : state.prepared;
  const std::uint32_t current =
      static_cast<std::uint32_t>(state.attempt_generation);
  const std::uint32_t seed =
      state.transactional ? current - std::uint32_t{1u} : current;
  const bool rebased = ops != nullptr &&
                       ops->seed_pipeline_generation != nullptr &&
                       ops->seed_pipeline_generation(selected, seed).ok;
  if (rebased) {
    state.native_generation = state.attempt_generation;
    state.native_parity = state.attempt_parity;
  }
  state.control_poisoned = !rebased;
  return rebased;
}

bool snapshot_valid(
    const std::shared_ptr<StateSnapshotState> &snapshot) noexcept {
  // StateSnapshotState has no public construction or mutation path.  Its
  // payload is sealed before publication, while restore() performs the full
  // schema/hash validation at the trust boundary.  Keep ordinary immutable
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

bool latest_state_valid(
    const std::shared_ptr<PipelinePublicationState> &publication) noexcept {
  if (publication == nullptr) {
    return false;
  }
  std::lock_guard lock{publication->gate};
  return publication->device != nullptr && !publication->state_pairs.empty() &&
         static_cast<bool>(publication->fingerprint) &&
         !publication->device_lost;
}

std::uint64_t latest_state_generation(
    const std::shared_ptr<PipelinePublicationState> &publication) noexcept {
  if (publication == nullptr) {
    return 0u;
  }
  std::lock_guard lock{publication->gate};
  return publication->generation;
}

graph::Fingerprint latest_state_fingerprint(
    const std::shared_ptr<PipelinePublicationState> &publication) noexcept {
  if (publication == nullptr) {
    return {};
  }
  std::lock_guard lock{publication->gate};
  return publication->fingerprint;
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

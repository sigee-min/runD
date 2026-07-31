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

[[nodiscard]] BufferState *published_buffer(PipelineState &state,
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

} // namespace

Result<std::shared_ptr<StateSnapshotState>>
snapshot_pipeline_state(const std::shared_ptr<PipelineState> &state) noexcept {
  if (!valid_pipeline(state)) {
    return Result<std::shared_ptr<StateSnapshotState>>::fail(
        Reason::PipelineInvalid);
  }
  std::lock_guard pipeline_lock{state->gate};
  if (state->device_lost) {
    return Result<std::shared_ptr<StateSnapshotState>>::fail(
        Reason::DeviceLost);
  }
  if (!state->transactional || state->state_pairs.empty()) {
    return Result<std::shared_ptr<StateSnapshotState>>::fail(
        Reason::PipelineInvalid);
  }
  if (state->phase == PipelinePhase::Running) {
    return Result<std::shared_ptr<StateSnapshotState>>::fail(
        Reason::PipelineBusy);
  }
  try {
    auto snapshot = std::make_shared<StateSnapshotState>();
    snapshot->fingerprint = state->fingerprint;
    snapshot->generation = state->generation;
    if (state->state_pairs.size() > PipelineLeafCapacity) {
      return Result<std::shared_ptr<StateSnapshotState>>::fail(
          Reason::PipelineCapacity);
    }
    snapshot->fields.reserve(state->state_pairs.size());
    std::size_t bytes = 0u;
    std::array<BufferClaim, PipelineLeafCapacity> claim_storage{};
    for (std::size_t index = 0u; index < state->state_pairs.size(); ++index) {
      const PipelineStatePair &pair = state->state_pairs[index];
      std::size_t next = 0u;
      if (!size::add(bytes, pair.bytes, next)) {
        return Result<std::shared_ptr<StateSnapshotState>>::fail(
            Reason::BufferCapacity);
      }
      snapshot->fields.push_back(PipelineSnapshotField{
          .type = pair.type,
          .format = pair.format,
          .count = pair.count,
          .offset = bytes,
          .bytes = pair.bytes,
          .payload_hash = ::rund::node::hash_detail::kFnvOffset,
      });
      bytes = next;
      claim_storage[index] = BufferClaim{
          .buffer = published_buffer(*state, index), .write = false};
    }
    if (bytes != 0u) {
      // Every byte is filled by the following CPU copy or backend download.
      // Default-initialized std::byte[] avoids vector::resize's redundant
      // value-initialization pass without exposing the storage publicly.
      snapshot->bytes.reset(new std::byte[bytes]);
    }
    snapshot->byte_count = bytes;
    const std::span<const BufferClaim> claims{claim_storage.data(),
                                              state->state_pairs.size()};
    const Status claimed = acquire_claims(*state->device, claims);
    if (!claimed) {
      return Result<std::shared_ptr<StateSnapshotState>>::fail(
          claimed.reason());
    }
    ClaimGuard guard{*state->device, claims};
    std::array<DownloadRequest, PipelineLeafCapacity> download_storage{};
    std::size_t download_count = 0u;
    for (std::size_t index = 0u; index < claims.size(); ++index) {
      BufferState *const source = claims[index].buffer;
      PipelineSnapshotField &field = snapshot->fields[index];
      if (field.bytes == 0u) {
        continue;
      }
      void *const destination = snapshot->bytes.get() + field.offset;
      if (state->device->backend == Backend::Cpu) {
        const CpuBufferState *const cpu =
            source == nullptr ? nullptr : cpu_buffer(*source);
        if (cpu == nullptr || cpu->data == nullptr ||
            cpu->bytes < field.bytes) {
          return Result<std::shared_ptr<StateSnapshotState>>::fail(
              Reason::TransferInvalid);
        }
        field.payload_hash = ::rund::node::hash_detail::CopyHash(
            cpu->data.get(), destination, field.bytes);
      } else {
        if (source == nullptr) {
          return Result<std::shared_ptr<StateSnapshotState>>::fail(
              Reason::TransferInvalid);
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
    if (state->device->backend != Backend::Cpu) {
      if (!downloads.empty() &&
          (state->device->ops == nullptr ||
           state->device->ops->download_batch == nullptr)) {
        return Result<std::shared_ptr<StateSnapshotState>>::fail(
            Reason::TransferInvalid);
      }
      if (!downloads.empty()) {
        const DownloadResult transfer =
            state->device->ops->download_batch(*state->device, downloads);
        if (!transfer.status) {
          return Result<std::shared_ptr<StateSnapshotState>>::fail(
              transfer.status.reason());
        }
        if (!transfer.payload_hash_valid) {
          return Result<std::shared_ptr<StateSnapshotState>>::fail(
              Reason::TransferInvalid);
        }
        record_download(*state, bytes, transfer, downloads.size());
      }
    }
    snapshot->hash = snapshot_hash(*snapshot);
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

Status restore_pipeline_state(
    const std::shared_ptr<PipelineState> &state,
    const std::shared_ptr<StateSnapshotState> &snapshot) noexcept {
  if (!valid_pipeline(state) || snapshot == nullptr) {
    return Status::fail(Reason::PipelineInvalid);
  }
  std::lock_guard pipeline_lock{state->gate};
  if (state->device_lost) {
    return Status::fail(Reason::DeviceLost);
  }
  if (!state->transactional || state->state_pairs.empty() ||
      state->phase != PipelinePhase::Ready) {
    return Status::fail(
        state->phase == PipelinePhase::Running    ? Reason::PipelineBusy
        : state->phase == PipelinePhase::Poisoned ? Reason::PipelinePoisoned
                                                  : Reason::PipelineInvalid);
  }
  if (!valid_snapshot_layout(*snapshot) ||
      snapshot->fingerprint != state->fingerprint ||
      snapshot->fields.size() != state->state_pairs.size()) {
    return Status::fail(Reason::PipelineInvalid);
  }
  if (state->state_pairs.size() > PipelineLeafCapacity) {
    return Status::fail(Reason::PipelineCapacity);
  }
  std::array<BufferClaim, PipelineLeafCapacity * 2u> claim_storage{};
  for (std::size_t index = 0u; index < state->state_pairs.size(); ++index) {
    const PipelineStatePair &pair = state->state_pairs[index];
    const PipelineSnapshotField &field = snapshot->fields[index];
    if (pair.type != field.type || pair.format != field.format ||
        pair.count != field.count || pair.bytes != field.bytes) {
      return Status::fail(Reason::PipelineInvalid);
    }
    claim_storage[index * 2u] =
        BufferClaim{.buffer = pair.first.get(), .write = true};
    claim_storage[index * 2u + 1u] =
        BufferClaim{.buffer = pair.second.get(), .write = true};
  }
  const std::span<const BufferClaim> claims{claim_storage.data(),
                                            state->state_pairs.size() * 2u};
  const Status claimed = acquire_claims(*state->device, claims, false);
  if (!claimed) {
    return claimed;
  }
  ClaimGuard guard{*state->device, claims};
  Status restored = Status::success();
  std::size_t copied = 0u;
  if (state->device->backend == Backend::Cpu) {
    for (std::size_t field_index = 0u;
         restored && field_index < snapshot->fields.size(); ++field_index) {
      const PipelineSnapshotField &field = snapshot->fields[field_index];
      for (std::size_t copy = 0u; copy < 2u; ++copy) {
        BufferState *const destination = claims[field_index * 2u + copy].buffer;
        if (field.bytes == 0u) {
          continue;
        }
        const void *const source = snapshot->bytes.get() + field.offset;
        CpuBufferState *const cpu =
            destination == nullptr ? nullptr : cpu_buffer(*destination);
        if (cpu == nullptr || cpu->data == nullptr ||
            cpu->bytes < field.bytes) {
          restored = Status::fail(Reason::TransferInvalid);
          break;
        }
        std::memcpy(cpu->data.get(), source, field.bytes);
        copied = ::rund::detail::counter::SaturatingAdd(copied, field.bytes);
        record_transfer(*state->device, field.bytes);
      }
    }
  } else {
    std::array<UploadRequest, PipelineLeafCapacity * 2u> upload_storage{};
    std::size_t upload_count = 0u;
    for (std::size_t field_index = 0u; field_index < snapshot->fields.size();
         ++field_index) {
      const PipelineSnapshotField &field = snapshot->fields[field_index];
      const void *const source =
          field.bytes == 0u ? nullptr : snapshot->bytes.get() + field.offset;
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
        (state->device->ops == nullptr ||
         state->device->ops->upload_batch == nullptr)) {
      restored = Status::fail(Reason::TransferInvalid);
    }
    if (restored && !uploads.empty()) {
      const UploadResult transfer =
          state->device->ops->upload_batch(*state->device, uploads);
      restored = transfer.status;
      if (restored) {
        for (const UploadRequest upload : uploads) {
          copied = ::rund::detail::counter::SaturatingAdd(copied, upload.bytes);
        }
        record_upload(*state, copied, transfer);
      }
    }
  }
  if (restored) {
    restored = seed_pipeline_generations(*state, snapshot->generation);
  }
  if (!restored) {
    publish_claims(*state->device, claims, false, true);
    guard.dismiss();
    state->failure = restored.reason();
    state->phase = PipelinePhase::Poisoned;
    return restored;
  }
  {
    std::lock_guard claim_lock{state->device->claims->gate};
    for (const BufferClaim claim : claims) {
      claim.buffer->poisoned = false;
      ++claim.buffer->generation;
      claim.buffer->writer = false;
    }
  }
  guard.dismiss();
  state->parity = 0u;
  state->generation = snapshot->generation;
  state->failure = Reason::Ok;
  state->phase = PipelinePhase::Ready;
  state->device_lost = false;
  state->control_poisoned = false;
  state->stats.publication.generation = state->generation;
  state->stats.publication.restore_byte_count =
      ::rund::detail::counter::SaturatingAdd(
          state->stats.publication.restore_byte_count, copied);
  return Status::success();
}

Status seed_pipeline_generations(PipelineState &state,
                                 const std::uint64_t generation) noexcept {
  // Empty and CPU Pipelines intentionally own no native generation control.
  if (state.device->backend == Backend::Cpu || state.active_step_count == 0u) {
    return Status::success();
  }
  const DeviceOps *const ops = state.device->ops;
  if (ops == nullptr || ops->seed_pipeline_generation == nullptr) {
    return Status::fail(Reason::PipelineInvalid);
  }
  const std::uint32_t current = static_cast<std::uint32_t>(generation);
  const std::uint32_t primary_seed =
      state.transactional ? current - std::uint32_t{1u} : current;
  const rund::AccelCheck primary =
      ops->seed_pipeline_generation(state.prepared, primary_seed);
  const rund::AccelCheck alternate =
      primary.ok && state.transactional
          ? ops->seed_pipeline_generation(state.alternate_prepared, current)
          : primary;
  return alternate.ok ? Status::success()
                      : Status::fail(project_reason(alternate.reason,
                                                    Reason::PipelineInvalid));
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
      state.transactional && state.parity != 0u ? state.alternate_prepared
                                                : state.prepared;
  const std::uint32_t current = static_cast<std::uint32_t>(state.generation);
  const std::uint32_t seed =
      state.transactional ? current - std::uint32_t{1u} : current;
  const bool rebased = ops != nullptr &&
                       ops->seed_pipeline_generation != nullptr &&
                       ops->seed_pipeline_generation(selected, seed).ok;
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

} // namespace rund::compute::detail

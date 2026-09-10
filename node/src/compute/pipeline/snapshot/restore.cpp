#include "internal.hpp"

#include <rund/counter.hpp>

#include "../../backend.hpp"
#include "../claim.hpp"
#include "../local.hpp"
#include "../transfer.hpp"

#include <array>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <limits>

namespace rund::compute::detail::snapshot_detail {

Status restore_locked(PipelineState &state,
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
  if (!valid_layout(snapshot) ||
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
        BufferState *const destination =
            claims[field_index * 2u + copy].buffer;
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
    record_pipeline_transfer(state, copied);
  } else {
    std::array<UploadRequest, PipelineLeafCapacity * 2u> upload_storage{};
    std::size_t upload_count = 0u;
    for (std::size_t field_index = 0u; field_index < snapshot.fields.size();
         ++field_index) {
      const PipelineSnapshotField &field = snapshot.fields[field_index];
      const void *const source =
          field.bytes == 0u ? nullptr : snapshot.bytes.get() + field.offset;
      for (std::size_t copy = 0u; copy < 2u; ++copy) {
        BufferState *const destination =
            claims[field_index * 2u + copy].buffer;
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
          node::accel::detail::TransferCompletion::Complete,
          node::accel::detail::TransferAuthority::Shared);
      restored = transfer.status;
      if (restored) {
        for (const UploadRequest upload : uploads) {
          copied = ::rund::detail::counter::SaturatingAdd(copied, upload.bytes);
        }
        record_pipeline_upload(state, copied, transfer);
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

} // namespace rund::compute::detail::snapshot_detail

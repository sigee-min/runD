#include "../batch.hpp"

#include "../../claim.hpp"
#include "../../residency/authority.hpp"
#include "../../transfer.hpp"
#include "model.hpp"

#include <array>
#include <cstddef>
#include <cstdint>
#include <cstring>

namespace rund::compute::detail {
namespace {

[[nodiscard]] bool write_attempted(const UploadResult &transfer) noexcept {
  return transfer.staging_bytes != 0u || transfer.staging_peak_bytes != 0u ||
         transfer.staging_reused_bytes != 0u ||
         transfer.buffer_allocations != 0u || transfer.buffer_reuses != 0u ||
         transfer.command_submits != 0u;
}

} // namespace

PipelineSlotUploadResult upload_pipeline_slots_impl(
    PipelineState &state, const std::span<const PipelineSlotUpload> slots,
    const node::accel::detail::TransferAuthority authority) noexcept {
  PipelineSlotUploadResult result{};
  const Status ready = validate_pipeline_transfer_ready(state);
  if (!ready) {
    result.transfer.status = ready;
    return result;
  }
  if (slots.size() > PipelineLeafCapacity || slots.empty()) {
    result.transfer.status = Status::fail(
        slots.empty() ? Reason::TransferInvalid : Reason::PipelineCapacity);
    return result;
  }
  const bool owner_local =
      authority == node::accel::detail::TransferAuthority::PipelinePrivate;
  if (owner_local &&
      (!has_private_residency_authority(state) || slots.size() != 1u ||
       state.residency_input >= state.resources.size() ||
       state.resources[state.residency_input].buffer.get() !=
           slots.front().buffer)) {
    result.transfer.status = Status::fail(Reason::TransferInvalid);
    return result;
  }
  std::array<BufferClaim, PipelineLeafCapacity> claim_storage{};
  std::array<UploadRequest, PipelineLeafCapacity> request_storage{};
  std::size_t request_count = 0u;
  for (std::size_t index = 0u; index < slots.size(); ++index) {
    const PipelineSlotUpload slot = slots[index];
    if (slot.buffer == nullptr || slot.buffer->device != state.device ||
        slot.bytes != slot.buffer->bytes ||
        (slot.bytes != 0u && slot.data == nullptr) ||
        !add_pipeline_transfer_bytes(slot.bytes, result.bytes)) {
      result.transfer.status = Status::fail(Reason::TransferInvalid);
      return result;
    }
    for (std::size_t prior = 0u; prior < index; ++prior) {
      if (slots[prior].buffer == slot.buffer) {
        result.transfer.status = Status::fail(Reason::TransferInvalid);
        return result;
      }
    }
    claim_storage[index] = BufferClaim{.buffer = slot.buffer, .write = true};
    if (slot.bytes != 0u) {
      request_storage[request_count++] = UploadRequest{
          .buffer = slot.buffer, .data = slot.data, .bytes = slot.bytes};
    }
  }
  const std::span<const BufferClaim> claims{claim_storage.data(), slots.size()};
  const Status claimed =
      owner_local ? Status::success() : acquire_claims(*state.device, claims);
  if (!claimed) {
    result.transfer.status = claimed;
    result.bytes = 0u;
    return result;
  }
  ClaimGuard guard{owner_local ? nullptr : state.device.get(), claims};
  if (request_count != 0u) {
    if (state.device->backend == Backend::Cpu) {
      for (std::size_t index = 0u; index < request_count; ++index) {
        const UploadRequest request = request_storage[index];
        CpuBufferState *const storage = cpu_buffer(*request.buffer);
        if (storage == nullptr || storage->data == nullptr ||
            storage->bytes != request.bytes) {
          result.transfer.status = Status::fail(Reason::TransferInvalid);
          result.bytes = 0u;
          return result;
        }
        std::memcpy(storage->data.get(), request.data, request.bytes);
      }
    } else {
      if (state.device->ops == nullptr ||
          state.device->ops->upload_batch == nullptr) {
        result.transfer.status = Status::fail(Reason::TransferInvalid);
        result.bytes = 0u;
        return result;
      }
      const bool prepared_transfer =
          state.residency_transfer_prepared && request_count == 1u &&
          state.device->ops->upload_pipeline_transfer != nullptr &&
          state.residency_input < state.resources.size() &&
          state.resources[state.residency_input].buffer.get() ==
              request_storage[0].buffer;
      result.transfer =
          prepared_transfer
              ? state.device->ops->upload_pipeline_transfer(
                    state, request_storage[0].data, request_storage[0].bytes)
              : state.device->ops->upload_batch(
                    *state.device,
                    std::span<const UploadRequest>{request_storage.data(),
                                                   request_count},
                    node::accel::detail::TransferCompletion::Complete,
                    authority);
    }
  }
  record_pipeline_upload(state, result.bytes, result.transfer);
  const bool succeeded = static_cast<bool>(result.transfer.status);
  if (owner_local) {
    for (const BufferClaim claim : claims) {
      if (succeeded) {
        ++claim.buffer->generation;
      } else if (write_attempted(result.transfer)) {
        claim.buffer->poisoned = true;
      }
    }
  } else {
    publish_claims(*state.device, claims, succeeded,
                   !succeeded && write_attempted(result.transfer));
  }
  guard.dismiss();
  if (!succeeded) {
    result.bytes = 0u;
  } else {
    ::rund::detail::counter::Accumulate(state.stats.host_write_bytes,
                                        result.bytes);
    if (result.bytes != 0u &&
        state.samples != PipelineState::SampleState::Inactive) {
      state.samples = PipelineState::SampleState::Dirty;
    }
  }
  return result;
}

PipelineSlotUploadResult upload_pipeline_slots(
    PipelineState &state,
    const std::span<const PipelineSlotUpload> slots) noexcept {
  return upload_pipeline_slots_impl(
      state, slots, node::accel::detail::TransferAuthority::Shared);
}

PipelineSlotUploadResult upload_pipeline_private_slots(
    PipelineState &state,
    const std::span<const PipelineSlotUpload> slots) noexcept {
  return upload_pipeline_slots_impl(
      state, slots, node::accel::detail::TransferAuthority::PipelinePrivate);
}

} // namespace rund::compute::detail

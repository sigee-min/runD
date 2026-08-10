#include "../batch.hpp"

#include "../../../../hash/fnv.hpp"
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

[[nodiscard]] Status validate_output(const PipelineState &state,
                                     const std::size_t output,
                                     const BufferState *const buffer,
                                     const std::size_t bytes,
                                     const bool validate_generation) noexcept {
  if (output >= state.outputs.size()) {
    return Status::fail(Reason::ReadBufferMismatch);
  }
  const PipelineOutputState &observed = state.outputs[output];
  if (observed.resource >= state.resources.size()) {
    return Status::fail(Reason::PipelineInvalid);
  }
  const PipelineResource &resource = state.resources[observed.resource];
  if (resource.output != output || resource.buffer.get() != buffer ||
      buffer == nullptr || resource.bytes != bytes || buffer->bytes != bytes ||
      (validate_generation && buffer->generation != observed.generation)) {
    return Status::fail(Reason::ReadBufferMismatch);
  }
  return Status::success();
}

} // namespace

PipelineSlotDownloadResult download_pipeline_slots_impl(
    PipelineState &state, const std::span<const PipelineSlotDownload> slots,
    const node::accel::detail::TransferAuthority authority) noexcept {
  PipelineSlotDownloadResult result{};
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
       state.residency_output >= state.resources.size() ||
       state.resources[state.residency_output].buffer.get() !=
           slots.front().buffer)) {
    result.transfer.status = Status::fail(Reason::TransferInvalid);
    return result;
  }
  std::array<BufferClaim, PipelineLeafCapacity> claim_storage{};
  std::array<DownloadRequest, PipelineLeafCapacity> request_storage{};
  std::array<std::uint64_t, PipelineLeafCapacity> hashes{};
  std::size_t request_count = 0u;
  for (std::size_t index = 0u; index < slots.size(); ++index) {
    const PipelineSlotDownload slot = slots[index];
    const Status output =
        validate_output(state, slot.output, slot.buffer, slot.bytes, false);
    if (!output || (slot.bytes != 0u && slot.data == nullptr) ||
        !add_pipeline_transfer_bytes(slot.bytes, result.bytes)) {
      result.transfer.status =
          output ? Status::fail(Reason::TransferInvalid) : output;
      return result;
    }
    for (std::size_t prior = 0u; prior < index; ++prior) {
      if (slots[prior].buffer == slot.buffer ||
          slots[prior].output == slot.output) {
        result.transfer.status = Status::fail(Reason::TransferInvalid);
        return result;
      }
    }
    claim_storage[index] =
        BufferClaim{.buffer = const_cast<BufferState *>(slot.buffer)};
    hashes[index] = ::rund::node::hash_detail::ZeroHash(0u);
    if (slot.bytes != 0u) {
      request_storage[request_count++] = DownloadRequest{
          .buffer = slot.buffer,
          .data = slot.data,
          .bytes = slot.bytes,
          .payload_hash = &hashes[index],
      };
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
  // Generation cannot change while the read claims are live. Revalidate the
  // canonical output projection after claim acquisition to close the race
  // with a public Buffer write that completed between validation and claim.
  for (const PipelineSlotDownload slot : slots) {
    const Status output =
        validate_output(state, slot.output, slot.buffer, slot.bytes, true);
    if (!output) {
      result.transfer.status = output;
      result.bytes = 0u;
      return result;
    }
  }
  if (request_count != 0u) {
    if (state.device->backend == Backend::Cpu) {
      for (std::size_t index = 0u; index < request_count; ++index) {
        const DownloadRequest request = request_storage[index];
        const CpuBufferState *const storage = cpu_buffer(*request.buffer);
        if (storage == nullptr || storage->data == nullptr ||
            storage->bytes != request.bytes ||
            request.payload_hash == nullptr) {
          result.transfer.status = Status::fail(Reason::TransferInvalid);
          result.bytes = 0u;
          return result;
        }
        std::memcpy(request.data, storage->data.get(), request.bytes);
        *request.payload_hash =
            ::rund::node::hash_detail::HashBytes(request.data, request.bytes);
      }
      result.transfer.payload_hash_valid = true;
    } else {
      if (state.device->ops == nullptr ||
          state.device->ops->download_batch == nullptr) {
        result.transfer.status = Status::fail(Reason::TransferInvalid);
        result.bytes = 0u;
        return result;
      }
      const bool prepared_transfer =
          state.residency_transfer_prepared && request_count == 1u &&
          state.device->ops->download_pipeline_transfer != nullptr &&
          state.residency_output < state.resources.size() &&
          state.resources[state.residency_output].buffer.get() ==
              request_storage[0].buffer;
      result.transfer =
          prepared_transfer
              ? state.device->ops->download_pipeline_transfer(
                    state, request_storage[0].data, request_storage[0].bytes,
                    request_storage[0].payload_hash)
              : state.device->ops->download_batch(
                    *state.device,
                    std::span<const DownloadRequest>{request_storage.data(),
                                                     request_count},
                    authority);
    }
    if (result.transfer.status && !result.transfer.payload_hash_valid) {
      result.transfer.status = Status::fail(Reason::TransferInvalid);
    }
    record_pipeline_download(state, result.bytes, result.transfer,
                             request_count);
    if (!result.transfer.status) {
      result.bytes = 0u;
      return result;
    }
    result.events = request_count;
  }
  for (std::size_t index = 0u; index < slots.size(); ++index) {
    const Status published = publish_pipeline_output_observation(
        state, slots[index].output, hashes[index]);
    if (!published) {
      result.transfer.status = published;
      result.bytes = 0u;
      return result;
    }
  }
  if (result.bytes != 0u &&
      state.samples != PipelineState::SampleState::Inactive) {
    state.samples = PipelineState::SampleState::Dirty;
  }
  return result;
}

PipelineSlotDownloadResult download_pipeline_slots(
    PipelineState &state,
    const std::span<const PipelineSlotDownload> slots) noexcept {
  return download_pipeline_slots_impl(
      state, slots, node::accel::detail::TransferAuthority::Shared);
}

PipelineSlotDownloadResult download_pipeline_private_slots(
    PipelineState &state,
    const std::span<const PipelineSlotDownload> slots) noexcept {
  return download_pipeline_slots_impl(
      state, slots, node::accel::detail::TransferAuthority::PipelinePrivate);
}

} // namespace rund::compute::detail

#include "../batch.hpp"

#include "../../../backend.hpp"
#include "../../claim.hpp"
#include "../../residency/authority.hpp"
#include "../../transfer.hpp"
#include "model.hpp"

#include <algorithm>
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

PipelineFrameUploadResult upload_pipeline_frames_impl(
    PipelineState &state, const std::span<const PipelineFrameUpload> frames,
    const node::accel::detail::TransferAuthority authority) noexcept {
  PipelineFrameUploadResult result{};
  const Status ready = validate_pipeline_transfer_ready(state);
  if (!ready) {
    result.transfer.status = ready;
    return result;
  }
  if (frames.size() > PipelineLeafCapacity || frames.empty()) {
    result.transfer.status = Status::fail(
        frames.empty() ? Reason::TransferInvalid : Reason::PipelineCapacity);
    return result;
  }
  const bool owner_local =
      authority == node::accel::detail::TransferAuthority::PipelinePrivate;
  if (owner_local &&
      (!has_private_residency_authority(state) ||
       state.residency_input >= state.resources.size() ||
       std::any_of(frames.begin(), frames.end(), [&](const auto &frame) {
         return state.resources[state.residency_input].buffer.get() !=
                frame.buffer;
       }))) {
    result.transfer.status = Status::fail(Reason::TransferInvalid);
    return result;
  }
  std::array<BufferClaim, PipelineLeafCapacity> claim_storage{};
  std::array<UploadRequest, PipelineLeafCapacity> request_storage{};
  std::size_t request_count = 0u;
  for (std::size_t index = 0u; index < frames.size(); ++index) {
    const PipelineFrameUpload frame = frames[index];
    if (frame.buffer == nullptr || frame.buffer->device != state.device ||
        frame.offset > frame.buffer->bytes ||
        frame.bytes > frame.buffer->bytes - frame.offset ||
        (frame.bytes != 0u && frame.data == nullptr) ||
        !add_pipeline_transfer_bytes(frame.bytes, result.bytes)) {
      result.transfer.status = Status::fail(Reason::TransferInvalid);
      return result;
    }
    for (std::size_t prior = 0u; prior < index; ++prior) {
      const bool overlap =
          frames[prior].buffer == frame.buffer &&
          frame.offset < frames[prior].offset + frames[prior].bytes &&
          frames[prior].offset < frame.offset + frame.bytes;
      if ((!owner_local && frames[prior].buffer == frame.buffer) || overlap) {
        result.transfer.status = Status::fail(Reason::TransferInvalid);
        return result;
      }
    }
    claim_storage[index] = BufferClaim{.buffer = frame.buffer, .write = true};
    if (frame.bytes != 0u) {
      request_storage[request_count++] = UploadRequest{.buffer = frame.buffer,
                                                       .data = frame.data,
                                                       .bytes = frame.bytes,
                                                       .offset = frame.offset};
    }
  }
  const std::span<const BufferClaim> claims{claim_storage.data(),
                                            frames.size()};
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
            request.offset > storage->bytes ||
            request.bytes > storage->bytes - request.offset) {
          result.transfer.status = Status::fail(Reason::TransferInvalid);
          result.bytes = 0u;
          return result;
        }
        std::memcpy(storage->data.get() + request.offset, request.data,
                    request.bytes);
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
          request_storage[0].offset == 0u &&
          request_storage[0].bytes == request_storage[0].buffer->bytes &&
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

PipelineFrameUploadResult upload_pipeline_frames(
    PipelineState &state,
    const std::span<const PipelineFrameUpload> frames) noexcept {
  return upload_pipeline_frames_impl(
      state, frames, node::accel::detail::TransferAuthority::Shared);
}

PipelineFrameUploadResult upload_pipeline_private_frames(
    PipelineState &state,
    const std::span<const PipelineFrameUpload> frames) noexcept {
  return upload_pipeline_frames_impl(
      state, frames, node::accel::detail::TransferAuthority::PipelinePrivate);
}

} // namespace rund::compute::detail

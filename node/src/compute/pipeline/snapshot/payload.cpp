#include "internal.hpp"

#include "../../backend.hpp"
#include "../../../hash/fnv.hpp"
#include "../claim.hpp"
#include "../local.hpp"
#include "../transfer.hpp"

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>

namespace rund::compute::detail::snapshot_detail {
namespace {

[[nodiscard]] BufferState *published_buffer(PipelinePublicationState &state,
                                            const std::size_t index) noexcept {
  if (index >= state.state_pairs.size()) {
    return nullptr;
  }
  PipelineStatePair &pair = state.state_pairs[index];
  return (state.parity == 0u ? pair.first : pair.second).get();
}

} // namespace

Status capture_payload(PipelineState &state,
                       PipelinePublicationState &publication,
                       StateSnapshotState &snapshot,
                       std::uint64_t &transfer_count) noexcept {
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
    const DownloadResult transfer = state.device->ops->download_batch(
        *state.device, downloads,
        node::accel::detail::TransferAuthority::Shared);
    if (!transfer.status) {
      if (transfer.status.reason() == Reason::DeviceLost) {
        publication.device_lost = true;
      }
      return transfer.status;
    }
    if (!transfer.payload_hash_valid) {
      return Status::fail(Reason::TransferInvalid);
    }
    record_pipeline_download(state, snapshot.byte_count, transfer,
                             downloads.size());
    transfer_count = std::max<std::uint64_t>(1u, transfer.command_submits);
  }
  snapshot.hash = hash(snapshot);
  return Status::success();
}

} // namespace rund::compute::detail::snapshot_detail

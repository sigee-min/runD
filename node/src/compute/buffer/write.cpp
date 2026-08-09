#include "../backend.hpp"
#include "../device/state.hpp"
#include "../pipeline/claim.hpp"
#include "../status.hpp"
#include "../type.hpp"
#include <rund/counter.hpp>

#include <cstring>

namespace rund::compute::detail {

Status write_buffer_measured(const std::shared_ptr<BufferState> &buffer,
                             const HostView input, WriteStats &stats,
                             UploadResult &transfer) noexcept {
  transfer = UploadResult{};
  if (buffer == nullptr || buffer->device == nullptr ||
      buffer->type != input.type || buffer->count != input.count ||
      (input.data == nullptr && input.count != 0u)) {
    return Status::fail(Reason::ShapeMismatch);
  }
  const std::size_t bytes = input.count * type_bytes(input.type);
  const BufferClaim claim{buffer.get(), true};
  const Status claimed = acquire_claims(*buffer->device, {&claim, 1u});
  if (!claimed) {
    return claimed;
  }
  ClaimGuard claim_guard{*buffer->device, {&claim, 1u}};
  if (bytes == 0u) {
    publish_claims(*buffer->device, {&claim, 1u}, true, false);
    claim_guard.dismiss();
    return Status::success();
  }
  if (buffer->device->backend == Backend::Cpu) {
    CpuBufferState *const target = cpu_buffer(*buffer);
    if (target == nullptr || target->bytes != bytes) {
      return Status::fail(Reason::TransferInvalid);
    }
    std::memcpy(target->data.get(), input.data, bytes);
    ::rund::detail::counter::Accumulate(stats.copies, 1u);
    ::rund::detail::counter::Accumulate(stats.bytes, bytes);
    publish_claims(*buffer->device, {&claim, 1u}, true, false);
    claim_guard.dismiss();
    return Status::success();
  }
  if (buffer->device->ops == nullptr ||
      buffer->device->ops->upload == nullptr) {
    return Status::fail(Reason::TransferInvalid);
  }
  transfer =
      buffer->device->ops->upload(*buffer->device, *buffer, input.data, bytes);
  if (!transfer.status) {
    publish_claims(*buffer->device, {&claim, 1u}, false, true);
    claim_guard.dismiss();
    return transfer.status;
  }
  ::rund::detail::counter::Accumulate(stats.uploads, 1u);
  ::rund::detail::counter::Accumulate(stats.bytes, bytes);
  publish_claims(*buffer->device, {&claim, 1u}, true, false);
  claim_guard.dismiss();
  return Status::success();
}

Status write_buffer(const std::shared_ptr<BufferState> &buffer,
                    const HostView input, WriteStats &stats) noexcept {
  UploadResult transfer{};
  const Status written = write_buffer_measured(buffer, input, stats, transfer);
  if (written && input.count != 0u) {
    record_transfer(*buffer->device, input.count * type_bytes(input.type));
  }
  return written;
}

} // namespace rund::compute::detail

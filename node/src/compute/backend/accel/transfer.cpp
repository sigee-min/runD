#include "../../device/state.hpp"
#include "../../buffer/state.hpp"
#include "transfer.hpp"

#include "../../../accel/context/internal/support.hpp"
#include "../../../accel/context/local.hpp"
#include "../../../accel/context/transfer.hpp"
#include "../../status.hpp"

#include <accel/context/buffer.hpp>
#include <rund/counter.hpp>

#include <algorithm>
#include <array>

namespace rund::compute::detail::accel_backend {
namespace {

constexpr std::size_t TransferCapacity = PipelineTransferCapacity;

void add_download(DownloadResult &total, const DownloadResult &part) noexcept {
  ::rund::detail::counter::Accumulate(total.staging_bytes, part.staging_bytes);
  total.staging_peak_bytes =
      std::max(total.staging_peak_bytes, part.staging_peak_bytes);
  ::rund::detail::counter::Accumulate(total.staging_reused_bytes,
                                      part.staging_reused_bytes);
  ::rund::detail::counter::Accumulate(total.buffer_allocations,
                                      part.buffer_allocations);
  ::rund::detail::counter::Accumulate(total.buffer_reuses, part.buffer_reuses);
  ::rund::detail::counter::Accumulate(total.command_submits,
                                      part.command_submits);
  ::rund::detail::counter::Accumulate(total.readback_ns, part.readback_ns);
  total.staging_reused = total.staging_bytes != 0u &&
                         total.staging_reused_bytes == total.staging_bytes;
}

DownloadResult
scalar_downloads(DeviceState &device,
                 const std::span<const DownloadRequest> requests) {
  DownloadResult total{
      .status = Status::success(),
      .payload_hash_valid = true,
  };
  for (std::size_t index = 0u; index < requests.size(); ++index) {
    const DownloadRequest &request = requests[index];
    if (request.buffer == nullptr || request.payload_hash == nullptr ||
        (request.bytes != 0u && request.data == nullptr)) {
      node::accel::detail::MarkDownloadFailure(
          request.outcome,
          node::accel::detail::DownloadRangeState::FailedNoWrite);
      total.status = Status::fail(Reason::TransferInvalid);
      total.payload_hash_valid = false;
      total.first_failed = index;
      total.first_failed_valid = true;
      return total;
    }
    const DownloadResult part = download(device, *request.buffer, request.data,
                                         request.bytes, request.offset);
    add_download(total, part);
    if (!part.status || !part.payload_hash_valid) {
      node::accel::detail::MarkDownloadFailure(
          request.outcome,
          part.confirmed_bytes != 0u
              ? node::accel::detail::DownloadRangeState::FailedMayWrite
              : node::accel::detail::DownloadRangeState::FailedNoWrite,
          part.confirmed_bytes);
      total.status =
          part.status ? Status::fail(Reason::TransferInvalid) : part.status;
      total.payload_hash_valid = false;
      ::rund::detail::counter::Accumulate(total.confirmed_bytes,
                                          part.confirmed_bytes);
      total.first_failed = index;
      total.first_failed_valid = true;
      return total;
    }
    *request.payload_hash = part.payload_hash;
    node::accel::detail::MarkDownloadComplete(request.outcome, request.bytes,
                                              part.payload_hash,
                                              part.payload_hash_valid);
    ::rund::detail::counter::Accumulate(total.confirmed_bytes, request.bytes);
    ++total.ordered_prefix;
  }
  return total;
}

} // namespace

UploadResult upload(DeviceState &device, BufferState &buffer,
                    const void *const data, const std::size_t bytes) {
  const UploadRequest request{.buffer = &buffer, .data = data, .bytes = bytes};
  return upload_batch(device, std::span<const UploadRequest>{&request, 1u},
                      node::accel::detail::TransferCompletion::Queued,
                      node::accel::detail::TransferAuthority::Shared);
}

Status resolve_buffer(const DeviceState &device, const BufferState &buffer,
                      std::shared_ptr<void> &handle) {
  const AccelDeviceState *const accel = accel_device(device);
  const AccelBufferState *const storage = accel_buffer(buffer);
  if (accel == nullptr || storage == nullptr) {
    handle.reset();
    return Status::fail(Reason::BindingDeviceMismatch);
  }
  const node::accel::detail::ContextAdmission admission =
      node::accel::detail::AdmitContextForSupport(accel->context);
  const rund::AccelCheck check =
      admission.check.ok ? node::accel::detail::ValidateAccelBufferForSupport(
                               admission, storage->buffer, handle)
                         : admission.check;
  if (!check.ok) {
    handle.reset();
    return Status::fail(Reason::BindingDeviceMismatch);
  }
  return Status::success();
}

UploadResult
upload_batch(DeviceState &device, const std::span<const UploadRequest> requests,
             const node::accel::detail::TransferCompletion completion,
             const node::accel::detail::TransferAuthority authority) {
  const AccelDeviceState *const accel = accel_device(device);
  if (accel == nullptr || requests.empty()) {
    return UploadResult{.status = Status::fail(Reason::TransferInvalid)};
  }
  if (requests.size() > TransferCapacity) {
    return UploadResult{.status = Status::fail(Reason::PipelineCapacity)};
  }
  std::array<node::accel::detail::UploadEntry, TransferCapacity> transfers{};
  std::array<node::accel::detail::UploadRoute, TransferCapacity> routes{};
  for (std::size_t index = 0u; index < requests.size(); ++index) {
    const UploadRequest request = requests[index];
    AccelBufferState *const target =
        request.buffer == nullptr ? nullptr : accel_buffer(*request.buffer);
    if (target == nullptr || (request.bytes != 0u && request.data == nullptr)) {
      return UploadResult{.status = Status::fail(Reason::TransferInvalid)};
    }
    transfers[index] = node::accel::detail::UploadEntry{
        .buffer = &target->buffer,
        .data = request.data,
        .bytes = request.bytes,
        .offset = request.offset,
    };
  }
  const node::accel::detail::AccelTransfer transfer =
      node::accel::detail::UploadAccelBuffers(
          accel->context,
          std::span<const node::accel::detail::UploadEntry>{transfers.data(),
                                                            requests.size()},
          std::span<node::accel::detail::UploadRoute>{routes.data(),
                                                      requests.size()},
          completion, authority);
  return UploadResult{
      .status = transfer.check.ok
                    ? Status::success()
                    : Status::fail(project_reason(transfer.check.reason,
                                                  Reason::TransferInvalid)),
      .staging_bytes = transfer.staging_bytes,
      .staging_peak_bytes = transfer.staging_peak_bytes,
      .staging_reused_bytes = transfer.staging_reused_bytes,
      .staging_budget = transfer.staging_budget,
      .buffer_allocations = transfer.buffer_allocations,
      .buffer_reuses = transfer.buffer_reuses,
      .command_submits = transfer.command_submits,
  };
}

DownloadResult download(DeviceState &device, const BufferState &buffer,
                        void *const data, const std::size_t bytes,
                        const std::size_t offset) {
  const AccelDeviceState *const accel = accel_device(device);
  const AccelBufferState *const resident = accel_buffer(buffer);
  if (accel == nullptr || resident == nullptr) {
    return DownloadResult{.status = Status::fail(Reason::TransferInvalid)};
  }
  const node::accel::detail::AccelTransfer transfer =
      node::accel::detail::DownloadAccelBufferMeasured(
          accel->context, resident->buffer, data, bytes, offset, true);
  return DownloadResult{
      .status = transfer.check.ok
                    ? Status::success()
                    : Status::fail(project_reason(transfer.check.reason,
                                                  Reason::TransferInvalid)),
      .payload_hash = transfer.payload_hash,
      .staging_bytes = transfer.staging_bytes,
      .staging_peak_bytes = transfer.staging_peak_bytes,
      .staging_reused_bytes = transfer.staging_reused_bytes,
      .staging_budget = transfer.staging_budget,
      .buffer_allocations = transfer.buffer_allocations,
      .buffer_reuses = transfer.buffer_reuses,
      .command_submits = transfer.command_submits,
      .readback_ns = transfer.readback_ns,
      .confirmed_bytes = transfer.confirmed_bytes,
      .ordered_prefix = transfer.ordered_prefix,
      .first_failed = transfer.first_failed,
      .staging_reused = transfer.staging_reused,
      .payload_hash_valid = transfer.check.ok && transfer.payload_hash_valid,
      .first_failed_valid = transfer.first_failed_valid,
  };
}

DownloadResult
download_batch(DeviceState &device,
               const std::span<const DownloadRequest> requests,
               const node::accel::detail::TransferAuthority authority) {
  const AccelDeviceState *const accel = accel_device(device);
  if (accel == nullptr || requests.empty()) {
    return DownloadResult{.status = Status::fail(Reason::TransferInvalid)};
  }
  if (requests.size() > TransferCapacity) {
    return scalar_downloads(device, requests);
  }
  std::array<node::accel::detail::DownloadEntry, TransferCapacity> transfers{};
  std::array<node::accel::detail::DownloadRoute, TransferCapacity> routes{};
  for (std::size_t index = 0u; index < requests.size(); ++index) {
    const DownloadRequest request = requests[index];
    const AccelBufferState *const source =
        request.buffer == nullptr ? nullptr : accel_buffer(*request.buffer);
    transfers[index] = node::accel::detail::DownloadEntry{
        .buffer = source == nullptr ? nullptr : &source->buffer,
        .data = request.data,
        .bytes = request.bytes,
        .offset = request.offset,
        .payload_hash = request.payload_hash,
        .outcome = request.outcome,
    };
  }
  const node::accel::detail::AccelTransfer transfer =
      node::accel::detail::DownloadAccelBuffersMeasured(
          accel->context,
          std::span<const node::accel::detail::DownloadEntry>{transfers.data(),
                                                              requests.size()},
          std::span<node::accel::detail::DownloadRoute>{routes.data(),
                                                        requests.size()},
          authority);
  return DownloadResult{
      .status = transfer.check.ok
                    ? Status::success()
                    : Status::fail(project_reason(transfer.check.reason,
                                                  Reason::TransferInvalid)),
      .staging_bytes = transfer.staging_bytes,
      .staging_peak_bytes = transfer.staging_peak_bytes,
      .staging_reused_bytes = transfer.staging_reused_bytes,
      .staging_budget = transfer.staging_budget,
      .buffer_allocations = transfer.buffer_allocations,
      .buffer_reuses = transfer.buffer_reuses,
      .command_submits = transfer.command_submits,
      .readback_ns = transfer.readback_ns,
      .confirmed_bytes = transfer.confirmed_bytes,
      .ordered_prefix = transfer.ordered_prefix,
      .first_failed = transfer.first_failed,
      .staging_reused = transfer.staging_reused,
      .payload_hash_valid = transfer.check.ok && transfer.payload_hash_valid,
      .first_failed_valid = transfer.first_failed_valid,
  };
}

CopyResult copy_batch(DeviceState &device,
                      const std::span<const CopyRequest> requests,
                      const node::accel::detail::TransferAuthority authority) {
  const AccelDeviceState *const accel = accel_device(device);
  if (accel == nullptr || requests.empty()) {
    return CopyResult{.status = Status::fail(Reason::TransferInvalid)};
  }
  if (requests.size() > TransferCapacity) {
    return CopyResult{.status = Status::fail(Reason::PipelineCapacity)};
  }
  std::array<node::accel::detail::CopyEntry, TransferCapacity> transfers{};
  std::array<node::accel::detail::CopyRoute, TransferCapacity> routes{};
  for (std::size_t index = 0u; index < requests.size(); ++index) {
    const CopyRequest request = requests[index];
    const AccelBufferState *const source =
        request.source == nullptr ? nullptr : accel_buffer(*request.source);
    AccelBufferState *const target =
        request.target == nullptr ? nullptr : accel_buffer(*request.target);
    if (source == nullptr || target == nullptr) {
      return CopyResult{.status = Status::fail(Reason::TransferInvalid)};
    }
    transfers[index] = node::accel::detail::CopyEntry{
        .source = &source->buffer,
        .target = &target->buffer,
        .bytes = request.bytes,
        .source_offset = request.source_offset,
        .target_offset = request.target_offset,
    };
  }
  const node::accel::detail::AccelCopy copied =
      node::accel::detail::CopyAccelBuffers(
          accel->context,
          std::span<const node::accel::detail::CopyEntry>{transfers.data(),
                                                          requests.size()},
          std::span<node::accel::detail::CopyRoute>{routes.data(),
                                                    requests.size()},
          authority);
  return CopyResult{
      .status = copied.check.ok
                    ? Status::success()
                    : Status::fail(project_reason(copied.check.reason,
                                                  Reason::TransferInvalid)),
      .command_submits = copied.command_submits,
  };
}

BufferReadView host_read(const DeviceState &device,
                         const BufferState &buffer) noexcept {
  const AccelDeviceState *const accel = accel_device(device);
  const AccelBufferState *const storage = accel_buffer(buffer);
  if (accel == nullptr || storage == nullptr) {
    return {};
  }
  const node::accel::detail::AccelHostView view =
      node::accel::detail::ReadAccelBuffer(accel->context, storage->buffer);
  return !view || view.bytes != buffer.bytes
             ? BufferReadView{}
             : BufferReadView{.data = view.data,
                              .bytes = static_cast<std::size_t>(view.bytes)};
}

BufferWriteView host_write(const DeviceState &device,
                           const BufferState &buffer) noexcept {
  const AccelDeviceState *const accel = accel_device(device);
  const AccelBufferState *const storage = accel_buffer(buffer);
  if (accel == nullptr || storage == nullptr) {
    return {};
  }
  const node::accel::detail::AccelHostWriteView view =
      node::accel::detail::WriteAccelBuffer(accel->context, storage->buffer);
  return !view || view.bytes != buffer.bytes
             ? BufferWriteView{}
             : BufferWriteView{.data = view.data,
                               .bytes = static_cast<std::size_t>(view.bytes)};
}

} // namespace rund::compute::detail::accel_backend

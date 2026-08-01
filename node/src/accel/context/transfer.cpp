#include <accel/check.hpp>
#include <accel/context/buffer.hpp>
#include <accel/context/value.hpp>

#include "../backend/resource.hpp"
#include "../clock.hpp"
#include "local.hpp"
#include "transfer.hpp"

#include <cstdint>

namespace rund::node::accel {

detail::AccelTransfer detail::DownloadAccelBufferMeasured(
    const rund::AccelContext &context, const rund::AccelBuffer &buffer,
    void *const data, const std::uint64_t bytes, const std::uint64_t offset,
    const bool hash_payload) {
  const TransferAdmission admission = AdmitAccelBufferTransfer(context, buffer);
  if (!admission.check.ok) {
    return {.check = admission.check};
  }
  if (!rund::kernel::checked::add(offset, bytes) ||
      offset + bytes > admission.byte_extent) {
    return {.check = RejectAccelCheck("accel_buffer_download_overflow")};
  }
  const std::uint64_t readback_begin = MonotonicNanoseconds();
  const BackendDownload downloaded = DownloadBackendBuffer(
      admission.pick, admission.route.ref, admission.route.handle, data, bytes,
      offset, hash_payload);
  return {.check = TransferCheckFrom(downloaded.check,
                                     "accel_buffer_download_overflow"),
          .payload_hash = downloaded.payload_hash,
          .staging_bytes = downloaded.staging_bytes,
          .staging_peak_bytes = downloaded.staging_peak_bytes,
          .staging_reused_bytes = downloaded.staging_reused_bytes,
          .staging_budget = context.pick.caps.staging_bytes,
          .buffer_allocations = downloaded.buffer_allocations,
          .buffer_reuses = downloaded.buffer_reuses,
          .command_submits = downloaded.command_submits,
          .readback_ns = MonotonicNanoseconds() - readback_begin,
          .staging_reused = downloaded.staging_reused,
          .payload_hash_valid =
              downloaded.check.ok && downloaded.payload_hash_valid};
}

detail::AccelTransfer
detail::UploadAccelBuffers(const rund::AccelContext &context,
                           const std::span<const UploadEntry> requests,
                           const std::span<UploadRoute> routes,
                           const TransferCompletion completion) {
  if (requests.empty() || routes.size() < requests.size()) {
    return {.check = RejectAccelCheck("accel_context_buffer_invalid")};
  }
  const ContextTokenAdmission context_admission = AdmitContextToken(context);
  if (!context_admission.check.ok) {
    return {.check = context_admission.check};
  }
  const std::shared_ptr<PickToken> &pick = context_admission.token->pick;
  for (std::size_t index = 0u; index < requests.size(); ++index) {
    const UploadEntry &request = requests[index];
    if (request.buffer == nullptr ||
        (request.bytes != 0u && request.data == nullptr)) {
      return {.check = RejectAccelCheck("accel_context_buffer_invalid")};
    }
    const TransferAdmission admission =
        AdmitAccelBufferTransfer(context_admission, *request.buffer);
    if (!admission.check.ok) {
      return {.check = admission.check};
    }
    if (!rund::kernel::checked::add(request.offset, request.bytes) ||
        request.offset + request.bytes > admission.byte_extent) {
      return {.check = RejectAccelCheck("accel_buffer_upload_overflow")};
    }
    routes[index] = UploadRoute{
        .resident = admission.route.ref,
        .handle = admission.route.handle,
        .data = request.data,
        .bytes = request.bytes,
        .offset = request.offset,
    };
  }
  const BackendUpload uploaded =
      UploadBackendBuffers(pick, routes.first(requests.size()), completion);
  return {.check =
              TransferCheckFrom(uploaded.check, "accel_buffer_upload_overflow"),
          .staging_bytes = uploaded.staging_bytes,
          .staging_peak_bytes = uploaded.staging_peak_bytes,
          .staging_reused_bytes = uploaded.staging_reused_bytes,
          .staging_budget = context.pick.caps.staging_bytes,
          .buffer_allocations = uploaded.buffer_allocations,
          .buffer_reuses = uploaded.buffer_reuses,
          .command_submits = uploaded.command_submits,
          .staging_reused =
              uploaded.staging_bytes != 0u &&
              uploaded.staging_reused_bytes == uploaded.staging_bytes};
}

detail::AccelTransfer detail::DownloadAccelBuffersMeasured(
    const rund::AccelContext &context,
    const std::span<const DownloadEntry> requests,
    const std::span<DownloadRoute> routes) {
  if (requests.empty() || routes.size() < requests.size()) {
    return {.check = RejectAccelCheck("accel_context_buffer_invalid")};
  }
  const ContextTokenAdmission context_admission = AdmitContextToken(context);
  if (!context_admission.check.ok) {
    return {.check = context_admission.check};
  }
  const std::shared_ptr<PickToken> &pick = context_admission.token->pick;
  for (std::size_t index = 0u; index < requests.size(); ++index) {
    const DownloadEntry &request = requests[index];
    if (request.buffer == nullptr || request.payload_hash == nullptr ||
        (request.bytes != 0u && request.data == nullptr)) {
      return {.check = RejectAccelCheck("accel_context_buffer_invalid")};
    }
    const TransferAdmission admission =
        AdmitAccelBufferTransfer(context_admission, *request.buffer);
    if (!admission.check.ok) {
      return {.check = admission.check};
    }
    if (!rund::kernel::checked::add(request.offset, request.bytes) ||
        request.offset + request.bytes > admission.byte_extent) {
      return {.check = RejectAccelCheck("accel_buffer_download_overflow")};
    }
    routes[index] = DownloadRoute{
        .resident = admission.route.ref,
        .handle = admission.route.handle,
        .data = request.data,
        .bytes = request.bytes,
        .offset = request.offset,
        .payload_hash = request.payload_hash,
    };
  }
  const std::uint64_t readback_begin = MonotonicNanoseconds();
  const BackendDownload downloaded =
      DownloadBackendBuffers(pick, routes.first(requests.size()));
  return {.check = TransferCheckFrom(downloaded.check,
                                     "accel_buffer_download_overflow"),
          .staging_bytes = downloaded.staging_bytes,
          .staging_peak_bytes = downloaded.staging_peak_bytes,
          .staging_reused_bytes = downloaded.staging_reused_bytes,
          .staging_budget = context.pick.caps.staging_bytes,
          .buffer_allocations = downloaded.buffer_allocations,
          .buffer_reuses = downloaded.buffer_reuses,
          .command_submits = downloaded.command_submits,
          .readback_ns = MonotonicNanoseconds() - readback_begin,
          .staging_reused = downloaded.staging_reused,
          .payload_hash_valid =
              downloaded.check.ok && downloaded.payload_hash_valid};
}

detail::AccelCopy
detail::CopyAccelBuffers(const rund::AccelContext &context,
                         const std::span<const CopyEntry> requests,
                         const std::span<CopyRoute> routes) {
  if (requests.empty() || routes.size() < requests.size()) {
    return {.check = RejectAccelCheck("accel_context_buffer_invalid")};
  }
  const ContextTokenAdmission context_admission = AdmitContextToken(context);
  if (!context_admission.check.ok) {
    return {.check = context_admission.check};
  }
  const std::shared_ptr<PickToken> &pick = context_admission.token->pick;
  for (std::size_t index = 0u; index < requests.size(); ++index) {
    const CopyEntry &request = requests[index];
    if (request.source == nullptr || request.target == nullptr) {
      return {.check = RejectAccelCheck("accel_context_buffer_invalid")};
    }
    const TransferAdmission source =
        AdmitAccelBufferTransfer(context_admission, *request.source);
    const TransferAdmission target =
        AdmitAccelBufferTransfer(context_admission, *request.target);
    if (!source.check.ok) {
      return {.check = source.check};
    }
    if (!target.check.ok) {
      return {.check = target.check};
    }
    if (!rund::kernel::checked::add(request.source_offset, request.bytes) ||
        request.source_offset + request.bytes > source.byte_extent ||
        !rund::kernel::checked::add(request.target_offset, request.bytes) ||
        request.target_offset + request.bytes > target.byte_extent) {
      return {.check = RejectAccelCheck("accel_buffer_copy_overflow")};
    }
    routes[index] = CopyRoute{
        .source = source.route.ref,
        .source_handle = source.route.handle,
        .target = target.route.ref,
        .target_handle = target.route.handle,
        .bytes = request.bytes,
        .source_offset = request.source_offset,
        .target_offset = request.target_offset,
    };
  }
  const BackendCopy copied =
      CopyBackendBuffers(pick, routes.first(requests.size()));
  return {.check =
              TransferCheckFrom(copied.check, "accel_buffer_copy_overflow"),
          .command_submits = copied.command_submits};
}

rund::AccelCheck UploadAccelBuffer(const rund::AccelContext &context,
                                   const rund::AccelBuffer &buffer,
                                   const void *const data,
                                   const std::uint64_t bytes,
                                   const std::uint64_t offset) {
  const detail::TransferAdmission admission =
      detail::AdmitAccelBufferTransfer(context, buffer);
  if (!admission.check.ok) {
    return admission.check;
  }
  if (!rund::kernel::checked::add(offset, bytes) ||
      offset + bytes > admission.byte_extent) {
    return detail::RejectAccelCheck("accel_buffer_upload_overflow");
  }
  return detail::TransferCheckFrom(
      detail::UploadBackendBuffer(admission.pick, admission.route.ref,
                                  admission.route.handle, data, bytes, offset),
      "accel_buffer_upload_overflow");
}

rund::AccelCheck DownloadAccelBuffer(const rund::AccelContext &context,
                                     const rund::AccelBuffer &buffer,
                                     void *const data,
                                     const std::uint64_t bytes,
                                     const std::uint64_t offset) {
  return detail::DownloadAccelBufferMeasured(context, buffer, data, bytes,
                                             offset)
      .check;
}

} // namespace rund::node::accel

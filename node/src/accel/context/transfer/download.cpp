#include <accel/check.hpp>
#include <accel/context/buffer.hpp>
#include <accel/context/value.hpp>

#include "../../backend/resource.hpp"
#include "../../clock.hpp"
#include "../../kernel/prepared/interface/api.hpp"
#include "../local.hpp"
#include "../transfer.hpp"

#include <cstdint>
#include <memory>

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
          .confirmed_bytes = downloaded.confirmed_bytes,
          .ordered_prefix = downloaded.ordered_prefix,
          .first_failed = downloaded.first_failed,
          .staging_reused = downloaded.staging_reused,
          .payload_hash_valid =
              downloaded.check.ok && downloaded.payload_hash_valid,
          .first_failed_valid = downloaded.first_failed_valid};
}

detail::AccelTransfer detail::DownloadAccelBuffersMeasured(
    const rund::AccelContext &context,
    const std::span<const DownloadEntry> requests,
    const std::span<DownloadRoute> routes, const TransferAuthority authority) {
  if (requests.empty() || routes.size() < requests.size()) {
    return {.check = RejectAccelCheck("accel_context_buffer_invalid")};
  }
  const std::shared_ptr<ContextToken> context_token =
      AdmitContextToken(context);
  if (context_token == nullptr) {
    return {.check = RejectAccelCheck("accel_context_buffer_invalid")};
  }
  const std::shared_ptr<PickToken> &pick = context_token->pick;
  for (const DownloadEntry &request : requests) {
    ResetDownloadOutcome(request.outcome);
  }
  std::size_t route_count = 0u;
  for (std::size_t index = 0u; index < requests.size(); ++index) {
    const DownloadEntry &request = requests[index];
    if (request.buffer == nullptr || request.payload_hash == nullptr ||
        (request.bytes != 0u && request.data == nullptr)) {
      routes[index] = DownloadRoute{
          .outcome = request.outcome,
          .failure_reason = "accel_context_buffer_invalid",
      };
      route_count = index + 1u;
      break;
    }
    const TransferAdmission admission =
        authority == TransferAuthority::PipelinePrivate
            ? AdmitAccelBufferPrivateTransfer(context_token, *request.buffer)
            : AdmitAccelBufferTransfer(context_token, *request.buffer);
    if (!admission.check.ok) {
      routes[index] = DownloadRoute{
          .outcome = request.outcome,
          .failure_reason = admission.check.reason,
      };
      route_count = index + 1u;
      break;
    }
    if (!rund::kernel::checked::add(request.offset, request.bytes) ||
        request.offset + request.bytes > admission.byte_extent) {
      routes[index] = DownloadRoute{
          .outcome = request.outcome,
          .failure_reason = "accel_buffer_download_overflow",
      };
      route_count = index + 1u;
      break;
    }
    routes[index] = DownloadRoute{
        .resident = admission.route.ref,
        .handle = admission.route.handle,
        .data = request.data,
        .bytes = request.bytes,
        .offset = request.offset,
        .payload_hash = request.payload_hash,
        .outcome = request.outcome,
    };
    route_count = index + 1u;
  }
  const std::uint64_t readback_begin = MonotonicNanoseconds();
  const BackendDownload downloaded =
      DownloadBackendBuffers(pick, routes.first(route_count), authority);
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
          .confirmed_bytes = downloaded.confirmed_bytes,
          .ordered_prefix = downloaded.ordered_prefix,
          .first_failed = downloaded.first_failed,
          .staging_reused = downloaded.staging_reused,
          .payload_hash_valid =
              downloaded.check.ok && downloaded.payload_hash_valid,
          .first_failed_valid = downloaded.first_failed_valid};
}

} // namespace rund::node::accel

#include <accel/check.hpp>
#include <accel/context/buffer.hpp>
#include <accel/context/value.hpp>

#include "../../backend/resource.hpp"
#include "../../kernel/prepared/interface/api.hpp"
#include "../local.hpp"
#include "../transfer.hpp"

#include <cstdint>
#include <memory>

namespace rund::node::accel {

detail::AccelTransfer
detail::UploadAccelBuffers(const rund::AccelContext &context,
                           const std::span<const UploadEntry> requests,
                           const std::span<UploadRoute> routes,
                           const TransferCompletion completion,
                           const TransferAuthority authority) {
  if (requests.empty() || routes.size() < requests.size()) {
    return {.check = RejectAccelCheck("accel_context_buffer_invalid")};
  }
  const std::shared_ptr<ContextToken> context_token =
      AdmitContextToken(context);
  if (context_token == nullptr) {
    return {.check = RejectAccelCheck("accel_context_buffer_invalid")};
  }
  const std::shared_ptr<PickToken> &pick = context_token->pick;
  for (std::size_t index = 0u; index < requests.size(); ++index) {
    const UploadEntry &request = requests[index];
    if (request.buffer == nullptr ||
        (request.bytes != 0u && request.data == nullptr)) {
      return {.check = RejectAccelCheck("accel_context_buffer_invalid")};
    }
    const TransferAdmission admission =
        authority == TransferAuthority::PipelinePrivate
            ? AdmitAccelBufferPrivateTransfer(context_token, *request.buffer)
            : AdmitAccelBufferTransfer(context_token, *request.buffer);
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
  const BackendUpload uploaded = UploadBackendBuffers(
      pick, routes.first(requests.size()), completion, authority);
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

} // namespace rund::node::accel

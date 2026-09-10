#include "../resource.hpp"

#include "local.hpp"

namespace rund::node::accel::detail {

rund::AccelCheck UploadBackendBuffer(const std::shared_ptr<PickToken> &token,
                                     const rund::Buffer &buffer,
                                     const void *const data,
                                     const std::uint64_t bytes,
                                     const std::uint64_t offset) {
  const rund::AccelCheck check = resource_detail::Validate(
      token, buffer, data, bytes, offset, "accel_buffer_upload_overflow");
  if (!check.ok || token->ops->upload == nullptr) {
    return check.ok
               ? rund::AccelCheck{false, "accel_buffer_backend_unavailable"}
               : check;
  }
  return UploadBackendBuffer(token, resource_detail::Resident(buffer),
                             buffer.handle, data, bytes, offset);
}

rund::AccelCheck
UploadBackendBuffer(const std::shared_ptr<PickToken> &token,
                    const rund::kernel::ResidentBufferRef &resident,
                    const std::shared_ptr<void> &handle, const void *const data,
                    const std::uint64_t bytes, const std::uint64_t offset) {
  if (!resource_detail::ValidRoute(token) || token->ops->upload == nullptr ||
      resident.id == 0u || resident.bytes == 0u || handle == nullptr ||
      (bytes != 0u && data == nullptr) || offset > resident.bytes ||
      bytes > resident.bytes - offset) {
    return rund::AccelCheck{false, "accel_buffer_backend_unavailable"};
  }
  return token->ops->upload(token->raw, resident, handle, data, bytes, offset);
}

BackendUpload UploadBackendBuffers(const std::shared_ptr<PickToken> &token,
                                   const std::span<const UploadRoute> requests,
                                   const TransferCompletion completion,
                                   const TransferAuthority authority) {
  if (!resource_detail::ValidRoute(token) || requests.empty()) {
    return {};
  }
  if (token->ops->upload_batch != nullptr) {
    return token->ops->upload_batch(token->raw, requests, completion,
                                    authority);
  }
  if (authority == TransferAuthority::PipelinePrivate ||
      token->ops->upload == nullptr) {
    return {};
  }
  for (const UploadRoute &request : requests) {
    const rund::AccelCheck uploaded =
        token->ops->upload(token->raw, request.resident, request.handle,
                           request.data, request.bytes, request.offset);
    if (!uploaded.ok) {
      return BackendUpload{.check = uploaded};
    }
  }
  return BackendUpload{.check = {true, "ok"}};
}

} // namespace rund::node::accel::detail

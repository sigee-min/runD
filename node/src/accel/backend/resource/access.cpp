#include "../resource.hpp"

#include "local.hpp"

namespace rund::node::accel::detail {

BackendCopy CopyBackendBuffers(const std::shared_ptr<PickToken> &token,
                               const std::span<const CopyRoute> requests,
                               const TransferAuthority authority) {
  if (!resource_detail::ValidRoute(token) || requests.empty() ||
      token->ops->copy_batch == nullptr) {
    return {};
  }
  return token->ops->copy_batch(token->raw, requests, authority);
}

BackendLookup
LookupBackendBuffer(const std::shared_ptr<PickToken> &token,
                    const rund::kernel::ResidentBufferRef &requested,
                    const std::shared_ptr<void> &handle) {
  if (!resource_detail::ValidRoute(token) || token->ops->lookup == nullptr) {
    return {};
  }
  return token->ops->lookup(token->raw, requested, handle);
}

BackendHostView
ReadBackendBuffer(const std::shared_ptr<PickToken> &token,
                  const rund::kernel::ResidentBufferRef &requested,
                  const std::shared_ptr<void> &handle) noexcept {
  if (!resource_detail::ValidRoute(token) || token->ops->host_read == nullptr) {
    return {};
  }
  return token->ops->host_read(token->raw, requested, handle);
}

BackendHostWriteView
WriteBackendBuffer(const std::shared_ptr<PickToken> &token,
                   const rund::kernel::ResidentBufferRef &requested,
                   const std::shared_ptr<void> &handle) noexcept {
  if (!resource_detail::ValidRoute(token) ||
      token->ops->host_write == nullptr) {
    return {};
  }
  return token->ops->host_write(token->raw, requested, handle);
}

rund::RuntimeStats ReadBackendStats(const std::shared_ptr<PickToken> &token) {
  if (!resource_detail::ValidRoute(token) || token->ops->stats == nullptr) {
    return rund::RuntimeStats{
        .outcome = {.reason = "accel_buffer_backend_unavailable"}};
  }
  return token->ops->stats(token->raw);
}

void ResetBackendStats(const std::shared_ptr<PickToken> &token) noexcept {
  if (resource_detail::ValidRoute(token) && token->ops->reset != nullptr) {
    token->ops->reset(token->raw);
  }
}

rund::node::accel::AccelMemoryStats
ReadBackendMemory(const std::shared_ptr<PickToken> &token) noexcept {
  if (!resource_detail::ValidRoute(token) || token->ops->memory == nullptr) {
    return {};
  }
  return token->ops->memory(token->raw);
}

} // namespace rund::node::accel::detail

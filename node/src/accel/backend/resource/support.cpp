#include "local.hpp"

#include "../match.hpp"
#include "../usage.hpp"

namespace rund::node::accel::detail::resource_detail {

bool ValidRoute(const std::shared_ptr<PickToken> &token) noexcept {
  return token != nullptr && token->ops != nullptr && token->raw.check.ok &&
         token->raw.owner != nullptr && token->raw.api == token->ops->api;
}

rund::AccelCheck Validate(const std::shared_ptr<PickToken> &token,
                          const rund::Buffer &buffer, const void *const data,
                          const std::uint64_t bytes, const std::uint64_t offset,
                          const char *const overflow_reason) noexcept {
  if (!ValidRoute(token) || !buffer.check.ok || buffer.id == 0u ||
      buffer.bytes == 0u || buffer.owner == nullptr ||
      buffer.handle == nullptr || !KnownUsage(buffer.usage) ||
      (bytes != 0u && data == nullptr)) {
    return rund::AccelCheck{false, "accel_buffer_unavailable"};
  }
  if (buffer.owner.get() != static_cast<const void *>(token.get())) {
    return rund::AccelCheck{false, "accel_buffer_owner_mismatch"};
  }
  if (!SameOwner(buffer.owner, token)) {
    return rund::AccelCheck{false, "accel_buffer_backend_unavailable"};
  }
  if (offset > buffer.bytes || bytes > buffer.bytes - offset) {
    return rund::AccelCheck{false, overflow_reason};
  }
  return rund::AccelCheck{true, "ok"};
}

rund::kernel::ResidentBufferRef Resident(const rund::Buffer &buffer) noexcept {
  return rund::kernel::ResidentBufferRef{
      .id = buffer.id,
      .bytes = buffer.bytes,
      .element_bytes = buffer.element_bytes,
      .stride_bytes = buffer.stride_bytes,
      .count = buffer.count,
      .usage = ResidentUsage(buffer.usage),
  };
}

} // namespace rund::node::accel::detail::resource_detail

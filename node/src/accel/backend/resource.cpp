#include "resource.hpp"

#include "match.hpp"
#include "resource/local.hpp"

namespace rund::node::accel::detail {

rund::Buffer CreateBackendBuffer(
    const std::shared_ptr<PickToken> &token, const rund::BufferDesc &desc,
    const BackendBufferInitialization initialization,
    const BackendBufferMemory memory, const std::uint64_t exact_storage_bytes) {
  if (!resource_detail::ValidRoute(token) || token->ops->create == nullptr) {
    return rund::Buffer{
        .check = rund::AccelCheck{false, "accel_buffer_backend_unavailable"}};
  }
  rund::Buffer created = token->ops->create(token->raw, desc, initialization,
                                            memory, exact_storage_bytes);
  if (!created.check.ok) {
    return created;
  }
  if (created.id == 0u || created.bytes == 0u || created.handle == nullptr ||
      !SameObject(created.owner, token->raw.owner)) {
    return rund::Buffer{
        .check = rund::AccelCheck{false, "accel_buffer_backend_unavailable"}};
  }
  created.owner = PublicPickOwner(token);
  return created;
}

} // namespace rund::node::accel::detail

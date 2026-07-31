#include "buffer.hpp"

#include <utility>

namespace rund::node::accel::detail {

rund::Buffer MakeBuffer(const rund::AccelDevice &pick,
                        const rund::BufferDesc &desc,
                        const rund::AccelCheck &check,
                        const rund::kernel::ResidentBufferRef &resident,
                        std::shared_ptr<void> handle,
                        const std::uint64_t storage_bytes,
                        const bool storage_reused) noexcept {
  if (!check.ok) {
    return rund::Buffer{.check = check};
  }
  return rund::Buffer{
      .check = check,
      .id = resident.id,
      .bytes = resident.bytes,
      .element_bytes = resident.element_bytes,
      .stride_bytes = resident.stride_bytes,
      .count = resident.count,
      .storage_bytes = storage_bytes,
      .storage_reused = storage_reused,
      .usage = desc.usage,
      .owner = pick.owner,
      .handle = std::move(handle),
  };
}

} // namespace rund::node::accel::detail

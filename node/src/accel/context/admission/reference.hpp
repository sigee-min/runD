#pragma once

#include <accel/buffer.hpp>
#include <accel/context/buffer/descriptor.hpp>

#include "../../backend/usage.hpp"

namespace rund::node::accel::detail {

[[nodiscard]] inline rund::kernel::ResidentBufferRef
ResidentRefFrom(const rund::kernel::ResidentBufferRef &canonical,
                const rund::AccelBufferDesc &desc) {
  return rund::kernel::ResidentBufferRef{
      .id = canonical.id,
      .bytes = canonical.bytes,
      .offset_bytes = 0u,
      .element_bytes = desc.scalar_width_bytes,
      .stride_bytes = desc.scalar_width_bytes,
      .count = desc.count,
      .usage = ResidentUsage(desc.usage),
  };
}

[[nodiscard]] inline bool PublicBufferMatchesCanonical(
    const rund::Buffer &buffer,
    const rund::kernel::ResidentBufferRef &canonical) noexcept {
  return buffer.id == canonical.id && buffer.bytes == canonical.bytes &&
         buffer.element_bytes == canonical.element_bytes &&
         buffer.stride_bytes == canonical.stride_bytes &&
         buffer.count == canonical.count && KnownUsage(buffer.usage) &&
         ResidentUsage(buffer.usage) == canonical.usage;
}

[[nodiscard]] inline bool
SameResidentRef(const rund::kernel::ResidentBufferRef &lhs,
                const rund::kernel::ResidentBufferRef &rhs) noexcept {
  return lhs.id == rhs.id && lhs.bytes == rhs.bytes &&
         lhs.offset_bytes == rhs.offset_bytes &&
         lhs.element_bytes == rhs.element_bytes &&
         lhs.stride_bytes == rhs.stride_bytes && lhs.count == rhs.count &&
         lhs.usage == rhs.usage;
}

} // namespace rund::node::accel::detail

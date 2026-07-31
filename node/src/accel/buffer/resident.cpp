#include <accel/buffer.hpp>

#include "../backend/usage.hpp"

#include <kernel/core/checked.hpp>

namespace rund::node::accel {
namespace {

[[nodiscard]] rund::kernel::ResidentBufferRef
RefFrom(const rund::Buffer &buffer, const std::uint64_t element_bytes,
        const std::uint64_t count, const std::uint32_t usage) noexcept {
  return rund::kernel::ResidentBufferRef{
      .id = buffer.id,
      .bytes = buffer.bytes,
      .element_bytes = element_bytes,
      .stride_bytes = element_bytes,
      .count = count,
      .usage = usage,
  };
}

[[nodiscard]] rund::kernel::ResidentBufferRef
TypedRef(const rund::Buffer &buffer, const std::uint64_t element_bytes,
         const std::uint64_t count, const std::uint32_t usage) noexcept {
  std::uint64_t bytes = 0u;
  if (!buffer.check.ok || buffer.id == 0u || buffer.handle == nullptr ||
      element_bytes == 0u || count == 0u || !detail::KnownUsage(buffer.usage) ||
      !rund::kernel::checked::mul(element_bytes, count, bytes)) {
    return {};
  }
  return bytes <= buffer.bytes ? RefFrom(buffer, element_bytes, count, usage)
                               : rund::kernel::ResidentBufferRef{};
}

} // namespace

rund::kernel::ResidentBufferRef ResidentRef(const rund::Buffer &buffer) {
  if (!buffer.check.ok || buffer.id == 0u || buffer.handle == nullptr ||
      !detail::KnownUsage(buffer.usage)) {
    return {};
  }
  return RefFrom(buffer, buffer.element_bytes, buffer.count,
                 detail::ResidentUsage(buffer.usage));
}

rund::kernel::ResidentBufferRef ResidentRead(const rund::Buffer &buffer,
                                             const std::uint64_t element_bytes,
                                             const std::uint64_t count) {
  if (buffer.usage == rund::BufferUsage::WriteOnly) {
    return {};
  }
  return TypedRef(buffer, element_bytes, count,
                  rund::kernel::kResidentUsageRead);
}

rund::kernel::ResidentBufferRef ResidentWrite(const rund::Buffer &buffer,
                                              const std::uint64_t element_bytes,
                                              const std::uint64_t count) {
  if (buffer.usage == rund::BufferUsage::ReadOnly) {
    return {};
  }
  return TypedRef(buffer, element_bytes, count,
                  rund::kernel::kResidentUsageWrite);
}

std::shared_ptr<void> ResidentHandle(const rund::Buffer &buffer) {
  if (!buffer.check.ok || buffer.id == 0u) {
    return {};
  }
  return buffer.handle;
}

} // namespace rund::node::accel

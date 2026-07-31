#pragma once

#include "model.hpp"

#include <cstdint>

namespace rund::node::accel::detail {

[[nodiscard]] inline rund::kernel::ResidentBufferRef
RefFromDesc(const std::uint64_t id, const ResidentDesc &desc) noexcept {
  return rund::kernel::ResidentBufferRef{
      .id = id,
      .bytes = desc.bytes,
      .element_bytes = desc.element_bytes,
      .stride_bytes = desc.stride_bytes,
      .count = desc.count,
      .usage = desc.usage,
  };
}

template <typename Buffer>
[[nodiscard]] inline rund::kernel::ResidentBufferRef
RefFromResident(const Buffer &buffer) noexcept {
  return rund::kernel::ResidentBufferRef{
      .id = buffer.id,
      .bytes = buffer.bytes,
      .element_bytes = buffer.element_bytes,
      .stride_bytes = buffer.stride_bytes,
      .count = buffer.count,
      .usage = buffer.usage,
  };
}

} // namespace rund::node::accel::detail

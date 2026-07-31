#pragma once

#include <accel/graph/value.hpp>

namespace rund {

[[nodiscard]] constexpr AccelGraphBufferRef AccelRead(
    const AccelBuffer &buffer, const char *const name = nullptr,
    const GraphBufferVisibility visibility = GraphBufferVisibility::External,
    const rund::kernel::u64 logical_id = 0u) noexcept {
  return AccelGraphBufferRef{
      .buffer = &buffer,
      .shape =
          AccelBufferDesc{
              .scalar_width_bytes = buffer.scalar_width_bytes,
              .count = buffer.count,
              .usage = buffer.usage,
          },
      .logical_id = logical_id,
      .role = rund::kernel::BufferRole::Read,
      .binding_name = name,
      .visibility = visibility,
  };
}

[[nodiscard]] constexpr AccelGraphBufferRef AccelRead(
    const AccelBufferDesc shape, const char *const name = nullptr,
    const GraphBufferVisibility visibility = GraphBufferVisibility::External,
    const rund::kernel::u64 logical_id = 0u) noexcept {
  return AccelGraphBufferRef{
      .shape = shape,
      .logical_id = logical_id,
      .role = rund::kernel::BufferRole::Read,
      .binding_name = name,
      .visibility = visibility,
  };
}

} // namespace rund

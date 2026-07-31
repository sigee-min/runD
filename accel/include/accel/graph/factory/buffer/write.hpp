#pragma once

#include <accel/graph/value.hpp>

namespace rund {

[[nodiscard]] constexpr AccelGraphBufferRef AccelWrite(
    const AccelBuffer &buffer, const char *const name = nullptr,
    const GraphBufferVisibility visibility = GraphBufferVisibility::External,
    const rund::kernel::u64 logical_id = 0u,
    const rund::kernel::BufferInit init =
        rund::kernel::BufferInit::Preserve) noexcept {
  return AccelGraphBufferRef{
      .buffer = &buffer,
      .shape =
          AccelBufferDesc{
              .scalar_width_bytes = buffer.scalar_width_bytes,
              .count = buffer.count,
              .usage = buffer.usage,
          },
      .logical_id = logical_id,
      .role = rund::kernel::BufferRole::Write,
      .init = init,
      .binding_name = name,
      .visibility = visibility,
  };
}

[[nodiscard]] constexpr AccelGraphBufferRef AccelWrite(
    const AccelBufferDesc shape, const char *const name = nullptr,
    const GraphBufferVisibility visibility = GraphBufferVisibility::External,
    const rund::kernel::u64 logical_id = 0u,
    const rund::kernel::BufferInit init =
        rund::kernel::BufferInit::Preserve) noexcept {
  return AccelGraphBufferRef{
      .shape = shape,
      .logical_id = logical_id,
      .role = rund::kernel::BufferRole::Write,
      .init = init,
      .binding_name = name,
      .visibility = visibility,
  };
}

} // namespace rund

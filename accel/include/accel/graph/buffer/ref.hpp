#pragma once

#include <accel/context/buffer.hpp>
#include <accel/context/buffer/descriptor.hpp>
#include <accel/graph/visibility.hpp>
#include <kernel/program/compute/graph/schema.hpp>

namespace rund {

struct AccelGraphBufferRef {
  const AccelBuffer *buffer = nullptr;
  AccelBufferDesc shape{};
  rund::kernel::u64 logical_id = 0u;
  rund::kernel::BufferRole role = rund::kernel::BufferRole::Read;
  rund::kernel::BufferInit init = rund::kernel::BufferInit::Preserve;
  const char *binding_name = nullptr;
  GraphBufferVisibility visibility = GraphBufferVisibility::External;
};

[[nodiscard]] constexpr AccelBufferDesc
AccelGraphBufferShape(const AccelGraphBufferRef &ref) noexcept {
  return ref.buffer == nullptr
             ? ref.shape
             : AccelBufferDesc{
                   .scalar_width_bytes = ref.buffer->scalar_width_bytes,
                   .count = ref.buffer->count,
                   .usage = ref.buffer->usage,
               };
}

[[nodiscard]] constexpr bool
AccelGraphBufferShapeValid(const AccelGraphBufferRef &ref) noexcept {
  const AccelBufferDesc shape = AccelGraphBufferShape(ref);
  const bool known_usage = shape.usage == BufferUsage::ReadOnly ||
                           shape.usage == BufferUsage::WriteOnly ||
                           shape.usage == BufferUsage::ReadWrite;
  return shape.scalar_width_bytes != 0u && shape.count != 0u && known_usage;
}

} // namespace rund

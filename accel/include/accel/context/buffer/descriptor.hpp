#pragma once

#include <accel/buffer.hpp>

#include <cstdint>

namespace rund {

struct AccelBufferDesc {
  std::uint64_t scalar_width_bytes = 0u;
  std::uint64_t count = 0u;
  BufferUsage usage = BufferUsage::ReadWrite;
};

} // namespace rund

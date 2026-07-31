#pragma once

#include <accel/context/buffer.hpp>
#include <kernel/program/compute/graph/schema.hpp>

#include <cstdint>

namespace rund {

struct AccelRunBinding {
  const AccelBuffer *buffer = nullptr;
  rund::kernel::BufferRole role = rund::kernel::BufferRole::Read;
  std::uint64_t offset_bytes = 0u;
  std::uint64_t element_count = 0u;
  std::uint64_t stride_bytes = 0u;
  std::uint64_t element_bytes = 0u;
  std::uint64_t alignment = 0u;
};

} // namespace rund

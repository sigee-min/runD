#pragma once

#include <accel/buffer.hpp>
#include <accel/context/buffer/descriptor.hpp>

#include "policy.hpp"

#include <cstdint>

namespace node_accel_contract::collective {

[[nodiscard]] inline rund::AccelBufferDesc
BufferDesc(const rund::BufferUsage usage,
           const std::uint64_t scalar_width_bytes = 4u,
           const std::uint64_t count = 8u) noexcept {
  return rund::AccelBufferDesc{
      .scalar_width_bytes = scalar_width_bytes,
      .count = count,
      .usage = usage,
  };
}

} // namespace node_accel_contract::collective

#pragma once

#include <accel/buffer.hpp>
#include <accel/context/buffer.hpp>
#include <accel/context/buffer/descriptor.hpp>
#include <accel/context/value.hpp>

#include <node/accel/context.hpp>

#include "model.hpp"

namespace node_accel_contract::fusion {

[[nodiscard]] inline rund::AccelBufferDesc BufferDesc(
    const rund::BufferUsage usage = rund::BufferUsage::ReadWrite) noexcept {
  return rund::AccelBufferDesc{
      .scalar_width_bytes = 4u,
      .count = 8u,
      .usage = usage,
  };
}

[[nodiscard]] inline rund::AccelBuffer
MakeBuffer(const rund::AccelContext &context, const rund::BufferUsage usage) {
  return rund::node::accel::CreateAccelBuffer(context, BufferDesc(usage));
}

} // namespace node_accel_contract::fusion

#pragma once

#include <accel/api.hpp>
#include <accel/buffer.hpp>
#include <accel/context/buffer/descriptor.hpp>
#include <accel/device.hpp>

namespace node_accel_contract::context {

[[nodiscard]] inline rund::AccelPolicy
Policy(const rund::AccelApi api, const bool allow_fake = false) noexcept {
  return rund::AccelPolicy{
      .preferred = {api, rund::AccelApi::Auto, rund::AccelApi::Auto},
      .preferred_count = 1u,
      .allow_fake = allow_fake,
  };
}

[[nodiscard]] inline rund::BufferDesc BufferDesc(
    const rund::BufferUsage usage = rund::BufferUsage::ReadWrite) noexcept {
  return rund::BufferDesc{
      .bytes = 64u,
      .usage = usage,
      .alignment = 16u,
  };
}

[[nodiscard]] inline rund::AccelBufferDesc TypedDesc(
    const rund::BufferUsage usage = rund::BufferUsage::ReadWrite) noexcept {
  return rund::AccelBufferDesc{
      .scalar_width_bytes = 4u,
      .count = 8u,
      .usage = usage,
  };
}

} // namespace node_accel_contract::context

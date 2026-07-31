#pragma once

#include <accel/api.hpp>
#include <accel/buffer.hpp>
#include <accel/context/buffer/descriptor.hpp>
#include <accel/device.hpp>

#include <cstdint>

namespace node_accel_contract::kernel_case {

[[nodiscard]] inline rund::AccelPolicy
Policy(const rund::AccelApi api) noexcept {
  return rund::AccelPolicy{
      .preferred = {api, rund::AccelApi::Auto, rund::AccelApi::Auto},
      .preferred_count = 1u,
      .allow_fake = false,
  };
}

[[nodiscard]] inline rund::AccelBufferDesc
BufferDesc(const rund::BufferUsage usage = rund::BufferUsage::ReadWrite,
           const std::uint64_t count = 8u) noexcept {
  return rund::AccelBufferDesc{
      .scalar_width_bytes = 4u,
      .count = count,
      .usage = usage,
  };
}

} // namespace node_accel_contract::kernel_case

#pragma once

#include <accel/buffer.hpp>
#include <accel/check.hpp>
#include <accel/context/value.hpp>

#include "state.hpp"

#include <memory>

namespace node_accel_contract::context {

[[nodiscard]] inline rund::Buffer
SyntheticBufferFor(const rund::AccelContext &ctx) {
  return rund::Buffer{
      .check = rund::AccelCheck{true, "ok"},
      .id = 77u,
      .bytes = 64u,
      .element_bytes = 1u,
      .stride_bytes = 1u,
      .count = 64u,
      .usage = rund::BufferUsage::ReadWrite,
      .owner = ctx.owner,
      .handle = std::make_shared<int>(5),
  };
}

} // namespace node_accel_contract::context

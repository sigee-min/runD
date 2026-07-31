#pragma once

#include <accel/buffer.hpp>
#include <accel/context/buffer.hpp>
#include <accel/context/buffer/descriptor.hpp>
#include <accel/context/value.hpp>
#include <accel/device.hpp>

namespace node_accel_contract::context {

struct State {
  bool available = false;
  bool unavailable_ok = false;
  rund::AccelDevice pick{};
  rund::AccelContext context{};
  rund::AccelContext second{};
  rund::Buffer buffer{};
  rund::AccelBufferDesc typed_desc{};
  rund::AccelBuffer typed{};
  rund::AccelBuffer created{};
};

} // namespace node_accel_contract::context

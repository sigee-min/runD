#include <accel/buffer.hpp>
#include <accel/device.hpp>

#include "local.hpp"
#include <node/accel/buffer.hpp>

namespace node_accel_contract {

bool PublicBufferApiRejectsInvalidDescriptors() {
  namespace fix = node_accel_contract::buffer;
  static_assert(sizeof(rund::BufferUsage) == sizeof(std::uint8_t));
  static_assert(!fix::HasElementBytes<rund::BufferDesc>);
  static_assert(!fix::HasStrideBytes<rund::BufferDesc>);
  static_assert(!fix::HasCount<rund::BufferDesc>);

  const rund::AccelDevice unavailable{};
  rund::BufferDesc desc = fix::BufferDesc();

  desc.bytes = 0u;
  if (!fix::CheckReason(
          rund::node::accel::CreateBuffer(unavailable, desc).check,
          "accel_buffer_bytes_zero")) {
    return false;
  }

  desc = fix::BufferDesc();
  desc.usage = static_cast<rund::BufferUsage>(99u);
  return fix::CheckReason(
      rund::node::accel::CreateBuffer(unavailable, desc).check,
      "accel_buffer_usage_invalid");
}

} // namespace node_accel_contract

#pragma once

#include <accel/api.hpp>
#include <accel/buffer.hpp>
#include <accel/device.hpp>

#include <node/accel/buffer.hpp>

#include "alias.hpp"
#include "context.hpp"
#include "handle.hpp"
#include "range.hpp"

#include <array>

namespace node_accel_contract {

bool PublicBufferApiRejectsRangeAndOwnerFailures(
    const rund::AccelDevice &pick) {
  namespace fix = node_accel_contract::buffer;
  std::array<std::uint32_t, 4u> data{};
  if (!fix::RejectsMissingUpload(pick, data)) {
    return false;
  }
  if (!pick.check.ok || pick.api != rund::AccelApi::Metal) {
    return true;
  }
  const rund::Buffer buffer =
      rund::node::accel::CreateBuffer(pick, fix::BufferDesc());
  return buffer.check.ok && fix::RejectsRangeOverflow(pick, buffer, data) &&
         fix::RejectsContextForgery(pick, buffer, data) &&
         fix::RejectsUnavailableHandles(pick, buffer, data) &&
         fix::RejectsAliasForgery(pick, buffer, data);
}

} // namespace node_accel_contract

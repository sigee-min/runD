#pragma once

#include <accel/buffer.hpp>
#include <accel/device.hpp>

#include <node/accel/buffer.hpp>

#include "../local.hpp"

#include <array>
#include <cstdint>
#include <memory>
#include <string_view>

namespace node_accel_contract::buffer {

[[nodiscard]] inline bool
RejectsContextForgery(const rund::AccelDevice &pick, const rund::Buffer &buffer,
                      std::array<std::uint32_t, 4u> &data) {
  rund::AccelDevice wrong_pick = pick;
  wrong_pick.owner = std::make_shared<int>(7);
  if (!CheckReason(rund::node::accel::UploadBuffer(wrong_pick, buffer,
                                                   data.data(), sizeof(data)),
                   "accel_buffer_backend_unavailable")) {
    return false;
  }

  rund::Buffer wrong_buffer = buffer;
  wrong_buffer.owner = std::make_shared<int>(7);
  if (!CheckReason(rund::node::accel::UploadBuffer(pick, wrong_buffer,
                                                   data.data(), sizeof(data)),
                   "accel_buffer_owner_mismatch")) {
    return false;
  }

  rund::AccelDevice null_context = pick;
  null_context.backend.context = nullptr;
  if (!CheckReason(
          rund::node::accel::CreateBuffer(null_context, BufferDesc()).check,
          "accel_buffer_backend_unavailable") ||
      !CheckReason(rund::node::accel::UploadBuffer(null_context, buffer,
                                                   data.data(), sizeof(data)),
                   "accel_buffer_backend_unavailable") ||
      rund::node::accel::ReadRuntimeStats(null_context).ok ||
      std::string_view{
          rund::node::accel::ReadRuntimeStats(null_context).reason} !=
          "accel_buffer_backend_unavailable") {
    return false;
  }

  int foreign_context = 0;
  rund::AccelDevice mixed_pick = pick;
  mixed_pick.backend.context = &foreign_context;
  return CheckReason(rund::node::accel::DownloadBuffer(
                         mixed_pick, buffer, data.data(), sizeof(data)),
                     "accel_buffer_backend_unavailable") &&
         !rund::node::accel::ReadRuntimeStats(mixed_pick).ok;
}

} // namespace node_accel_contract::buffer

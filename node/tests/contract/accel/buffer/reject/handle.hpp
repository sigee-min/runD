#pragma once

#include <accel/buffer.hpp>
#include <accel/device.hpp>

#include <node/accel/buffer.hpp>

#include "../local.hpp"

#include <array>
#include <cstdint>
#include <memory>

namespace node_accel_contract::buffer {

[[nodiscard]] inline bool
RejectsUnavailableHandles(const rund::AccelDevice &pick,
                          const rund::Buffer &buffer,
                          std::array<std::uint32_t, 4u> &data) {
  rund::Buffer missing_handle = buffer;
  missing_handle.handle.reset();
  if (rund::node::accel::ResidentRef(missing_handle).id != 0u ||
      !CheckReason(rund::node::accel::UploadBuffer(pick, missing_handle,
                                                   data.data(), sizeof(data)),
                   "accel_buffer_unavailable") ||
      !CheckReason(rund::node::accel::DownloadBuffer(pick, missing_handle,
                                                     data.data(), sizeof(data)),
                   "accel_buffer_unavailable") ||
      !CheckReason(
          rund::node::accel::UploadBuffer(pick, missing_handle, nullptr, 0u),
          "accel_buffer_unavailable") ||
      !CheckReason(
          rund::node::accel::DownloadBuffer(pick, missing_handle, nullptr, 0u),
          "accel_buffer_unavailable")) {
    return false;
  }

  rund::Buffer wrong_handle = buffer;
  wrong_handle.handle = std::make_shared<int>(11);
  if (!CheckReason(rund::node::accel::UploadBuffer(pick, wrong_handle,
                                                   data.data(), sizeof(data)),
                   "accel_buffer_unavailable") ||
      !CheckReason(rund::node::accel::DownloadBuffer(pick, wrong_handle,
                                                     data.data(), sizeof(data)),
                   "accel_buffer_unavailable")) {
    return false;
  }

  const rund::Buffer copied = buffer;
  return rund::node::accel::UploadBuffer(pick, buffer, nullptr, 0u).ok &&
         rund::node::accel::DownloadBuffer(pick, buffer, nullptr, 0u).ok &&
         rund::node::accel::UploadBuffer(pick, copied, data.data(),
                                         sizeof(data))
             .ok &&
         rund::node::accel::DownloadBuffer(pick, copied, data.data(),
                                           sizeof(data))
             .ok &&
         rund::node::accel::UploadBuffer(pick, copied, nullptr, 0u).ok &&
         rund::node::accel::DownloadBuffer(pick, copied, nullptr, 0u).ok;
}

} // namespace node_accel_contract::buffer

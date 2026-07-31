#pragma once

#include <accel/buffer.hpp>
#include <accel/device.hpp>

#include <node/accel/buffer.hpp>

#include "../local.hpp"

#include <array>
#include <cstdint>

namespace node_accel_contract::buffer {

[[nodiscard]] inline bool
RejectsMissingUpload(const rund::AccelDevice &pick,
                     std::array<std::uint32_t, 4u> &data) {
  const rund::Buffer missing{};
  return CheckReason(
      rund::node::accel::UploadBuffer(pick, missing, data.data(), sizeof(data)),
      "accel_buffer_unavailable");
}

[[nodiscard]] inline bool
RejectsRangeOverflow(const rund::AccelDevice &pick, const rund::Buffer &buffer,
                     std::array<std::uint32_t, 4u> &data) {
  return CheckReason(rund::node::accel::UploadBuffer(pick, buffer, data.data(),
                                                     buffer.bytes + 1u),
                     "accel_buffer_upload_overflow") &&
         CheckReason(rund::node::accel::DownloadBuffer(
                         pick, buffer, data.data(), buffer.bytes + 1u),
                     "accel_buffer_download_overflow");
}

} // namespace node_accel_contract::buffer

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
RejectsAliasForgery(const rund::AccelDevice &pick, const rund::Buffer &buffer,
                    std::array<std::uint32_t, 4u> &data) {
  rund::AccelDevice alias_owner_pick = pick;
  alias_owner_pick.owner =
      std::shared_ptr<void>(pick.backend.context, [](void *) {});
  if (!CheckReason(
          rund::node::accel::CreateBuffer(alias_owner_pick, BufferDesc()).check,
          "accel_buffer_backend_unavailable") ||
      !CheckReason(rund::node::accel::UploadBuffer(alias_owner_pick, buffer,
                                                   data.data(), sizeof(data)),
                   "accel_buffer_backend_unavailable") ||
      rund::node::accel::ReadRuntimeStats(alias_owner_pick).ok ||
      std::string_view{
          rund::node::accel::ReadRuntimeStats(alias_owner_pick).reason} !=
          "accel_buffer_backend_unavailable") {
    return false;
  }

  rund::Buffer alias_handle = buffer;
  alias_handle.handle =
      std::shared_ptr<void>(buffer.handle.get(), [](void *) {});
  if (!CheckReason(rund::node::accel::UploadBuffer(pick, alias_handle,
                                                   data.data(), sizeof(data)),
                   "accel_buffer_unavailable") ||
      !CheckReason(rund::node::accel::DownloadBuffer(pick, alias_handle,
                                                     data.data(), sizeof(data)),
                   "accel_buffer_unavailable") ||
      !CheckReason(
          rund::node::accel::UploadBuffer(pick, alias_handle, nullptr, 0u),
          "accel_buffer_unavailable") ||
      !CheckReason(
          rund::node::accel::DownloadBuffer(pick, alias_handle, nullptr, 0u),
          "accel_buffer_unavailable")) {
    return false;
  }

  int alias_value{};
  alias_handle = buffer;
  alias_handle.handle = std::shared_ptr<void>(buffer.handle, &alias_value);
  return CheckReason(rund::node::accel::UploadBuffer(pick, alias_handle,
                                                     data.data(), sizeof(data)),
                     "accel_buffer_unavailable") &&
         CheckReason(rund::node::accel::DownloadBuffer(
                         pick, alias_handle, data.data(), sizeof(data)),
                     "accel_buffer_unavailable") &&
         CheckReason(
             rund::node::accel::UploadBuffer(pick, alias_handle, nullptr, 0u),
             "accel_buffer_unavailable") &&
         CheckReason(
             rund::node::accel::DownloadBuffer(pick, alias_handle, nullptr, 0u),
             "accel_buffer_unavailable");
}

} // namespace node_accel_contract::buffer

#include <accel/api.hpp>
#include <accel/buffer.hpp>
#include <accel/check.hpp>
#include <accel/device.hpp>
#include <accel/runtime.hpp>

#include "local.hpp"
#include <node/accel/buffer.hpp>

#include <array>
#include <cstdint>
#include <string_view>

namespace node_accel_contract {

bool PublicBufferApiExposesMetalResidencyWhenAvailable(
    const rund::AccelDevice &pick) {
  namespace fix = node_accel_contract::buffer;
  if (!pick.check.ok || pick.api != rund::AccelApi::Metal) {
    const rund::Buffer buffer =
        rund::node::accel::CreateBuffer(pick, fix::BufferDesc());
    return fix::CheckReason(buffer.check, "accel_buffer_backend_unavailable");
  }

  rund::BufferDesc desc = fix::BufferDesc();
  desc.usage = rund::BufferUsage::ReadOnly;
  const rund::Buffer read_buffer = rund::node::accel::CreateBuffer(pick, desc);
  if (!read_buffer.check.ok || read_buffer.id == 0u ||
      read_buffer.owner != pick.owner) {
    return false;
  }
  const rund::kernel::ResidentBufferRef read_ref =
      rund::node::accel::ResidentRef(read_buffer);
  if (read_ref.id != read_buffer.id || read_ref.bytes != desc.bytes ||
      read_ref.element_bytes != 1u || read_ref.stride_bytes != 1u ||
      read_ref.count != desc.bytes ||
      read_ref.usage != rund::kernel::kResidentUsageRead) {
    return false;
  }

  desc.usage = rund::BufferUsage::WriteOnly;
  const rund::Buffer write_buffer = rund::node::accel::CreateBuffer(pick, desc);
  const rund::kernel::ResidentBufferRef write_ref =
      rund::node::accel::ResidentRef(write_buffer);
  if (!write_buffer.check.ok ||
      write_ref.usage != rund::kernel::kResidentUsageWrite) {
    return false;
  }

  desc.usage = rund::BufferUsage::ReadWrite;
  const rund::Buffer read_write_buffer =
      rund::node::accel::CreateBuffer(pick, desc);
  const rund::kernel::ResidentBufferRef read_write_ref =
      rund::node::accel::ResidentRef(read_write_buffer);
  return read_write_buffer.check.ok &&
         read_write_ref.usage == rund::kernel::kResidentUsageWrite;
}

bool PublicBufferApiRoundTripsAndReportsStatsWhenAvailable(
    const rund::AccelDevice &pick) {
  namespace fix = node_accel_contract::buffer;
  if (!pick.check.ok || pick.api != rund::AccelApi::Metal) {
    const rund::RuntimeStats stats = rund::node::accel::ReadRuntimeStats(pick);
    return !stats.ok &&
           std::string_view{stats.reason} == "accel_buffer_backend_unavailable";
  }

  rund::node::accel::ResetRuntimeStats(pick);
  const rund::Buffer buffer =
      rund::node::accel::CreateBuffer(pick, fix::BufferDesc());
  if (!buffer.check.ok) {
    return false;
  }

  const std::array<std::uint32_t, 4u> input{3u, 5u, 8u, 13u};
  std::array<std::uint32_t, 4u> output{};
  const rund::AccelCheck upload = rund::node::accel::UploadBuffer(
      pick, buffer, input.data(), sizeof(input));
  const rund::AccelCheck download = rund::node::accel::DownloadBuffer(
      pick, buffer, output.data(), sizeof(output));
  const rund::RuntimeStats stats = rund::node::accel::ReadRuntimeStats(pick);

  return upload.ok && download.ok && input == output && stats.ok &&
         stats.buffer_allocation_count >= 1u &&
         stats.host_to_device_bytes == sizeof(input) &&
         stats.device_to_host_bytes == sizeof(output);
}

} // namespace node_accel_contract

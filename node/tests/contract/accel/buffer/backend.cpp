#include <accel/api.hpp>
#include <accel/buffer.hpp>
#include <accel/check.hpp>
#include <accel/device.hpp>
#include <accel/runtime.hpp>

#include "backend/id/high.hpp"
#include "local.hpp"
#include <array>
#include <cstdint>
#include <node/accel/buffer.hpp>
#include <node/accel/pick.hpp>
#include <string_view>

namespace node_accel_contract {
namespace {

[[nodiscard]] bool VulkanRoundTripsWhenAvailable() {
  namespace fix = node_accel_contract::buffer;
  const rund::BufferDesc desc = fix::BufferDesc();
  const rund::AccelDevice vulkan =
      rund::node::accel::PickAccel(fix::Policy(rund::AccelApi::Vulkan));
  if (!vulkan.check.ok) {
    return fix::VulkanPickReasonIsPrecise(vulkan);
  }

  const rund::Buffer buffer = rund::node::accel::CreateBuffer(vulkan, desc);
  std::array<std::uint32_t, 4u> input{2u, 4u, 6u, 8u};
  std::array<std::uint32_t, 4u> output{};
  rund::node::accel::ResetRuntimeStats(vulkan);
  const rund::AccelCheck upload = rund::node::accel::UploadBuffer(
      vulkan, buffer, input.data(), sizeof(input));
  const rund::AccelCheck download = rund::node::accel::DownloadBuffer(
      vulkan, buffer, output.data(), sizeof(output));
  const rund::RuntimeStats stats = rund::node::accel::ReadRuntimeStats(vulkan);
  return buffer.check.ok && upload.ok && download.ok && input == output &&
         stats.ok && stats.host_to_device_bytes == sizeof(input) &&
         stats.device_to_host_bytes == sizeof(output);
}

} // namespace

bool PublicBufferApiRejectsUnavailableBackends() {
  namespace fix = node_accel_contract::buffer;
  const rund::BufferDesc desc = fix::BufferDesc();
  const rund::AccelDevice unavailable{};
  if (!fix::CheckReason(
          rund::node::accel::CreateBuffer(unavailable, desc).check,
          "accel_buffer_backend_unavailable")) {
    return false;
  }

  const rund::AccelDevice fake =
      rund::node::accel::PickAccel(fix::Policy(rund::AccelApi::Fake, true));
  if (!fake.check.ok || fake.api != rund::AccelApi::Fake) {
    return false;
  }
  if (!fix::CheckReason(rund::node::accel::CreateBuffer(fake, desc).check,
                        "accel_buffer_backend_unavailable")) {
    return false;
  }

  const rund::RuntimeStats fake_stats =
      rund::node::accel::ReadRuntimeStats(fake);
  return !fake_stats.ok &&
         std::string_view{fake_stats.reason} ==
             "accel_buffer_backend_unavailable" &&
         VulkanRoundTripsWhenAvailable() &&
         fix::HighResidentIdsRoundTripAcrossBackends();
}

} // namespace node_accel_contract

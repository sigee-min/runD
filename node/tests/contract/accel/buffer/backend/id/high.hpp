#pragma once

#include <accel/api.hpp>
#include <accel/buffer.hpp>
#include <accel/device.hpp>

#include <node/accel/pick.hpp>

#include <node/accel/buffer.hpp>

#include "../../local.hpp"

#include <array>
#include <cstdint>

namespace node_accel_contract::buffer {

[[nodiscard]] inline bool
HighResidentIdRoundTripsWhenAvailable(const rund::AccelApi api) {
  const rund::AccelDevice pick = rund::node::accel::PickAccel(Policy(api));
  if (!pick.check.ok) {
    return api != rund::AccelApi::Cpu;
  }

  constexpr std::size_t kBufferCount = 33u;
  std::array<rund::Buffer, kBufferCount> buffers{};
  for (rund::Buffer &buffer : buffers) {
    buffer = rund::node::accel::CreateBuffer(pick, BufferDesc());
    if (!buffer.check.ok) {
      return false;
    }
  }
  if (rund::node::accel::ResidentRef(buffers.back()).id < kBufferCount) {
    return false;
  }

  const std::array<std::uint32_t, 4u> input{3u, 5u, 8u, 13u};
  std::array<std::uint32_t, 4u> output{};
  rund::Buffer zero = buffers.back();
  zero.id = 0u;
  rund::Buffer out_of_range = buffers.back();
  out_of_range.id += 1000000u;
  if (rund::node::accel::UploadBuffer(pick, zero, input.data(), sizeof(input))
          .ok ||
      rund::node::accel::DownloadBuffer(pick, out_of_range, output.data(),
                                        sizeof(output))
          .ok) {
    return false;
  }
  return rund::node::accel::UploadBuffer(pick, buffers.back(), input.data(),
                                         sizeof(input))
             .ok &&
         rund::node::accel::DownloadBuffer(pick, buffers.back(), output.data(),
                                           sizeof(output))
             .ok &&
         input == output;
}

[[nodiscard]] inline bool HighResidentIdsRoundTripAcrossBackends() {
  return HighResidentIdRoundTripsWhenAvailable(rund::AccelApi::Cpu) &&
         HighResidentIdRoundTripsWhenAvailable(rund::AccelApi::Metal) &&
         HighResidentIdRoundTripsWhenAvailable(rund::AccelApi::Vulkan);
}

} // namespace node_accel_contract::buffer

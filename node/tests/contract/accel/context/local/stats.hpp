#pragma once

#include <accel/runtime.hpp>

namespace node_accel_contract::context {

[[nodiscard]] inline bool
RuntimeBytesInclude(const rund::RuntimeStats &stats,
                    const std::uint64_t host_to_device,
                    const std::uint64_t device_to_host) noexcept {
  return stats.ok && stats.host_to_device_bytes == host_to_device &&
         stats.device_to_host_bytes == device_to_host;
}

} // namespace node_accel_contract::context

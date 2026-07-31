#pragma once

#include <rund/compute/telemetry.hpp>

#include <utility>

namespace rund::compute::detail {

struct ProfileAccess final {
  [[nodiscard]] static telemetry::Profile
  make(DeviceInfo device, const Stats execution, const MemoryStats memory) {
    return telemetry::Profile{std::move(device), execution, memory};
  }
};

} // namespace rund::compute::detail

#pragma once

#include <rund/compute/device/info.hpp>
#include <rund/compute/status.hpp>

#include <memory>

namespace rund::compute::detail {

struct DeviceState;

[[nodiscard]] Result<DeviceInfo>
snapshot_device_info(const std::shared_ptr<DeviceState> &state) noexcept;

} // namespace rund::compute::detail

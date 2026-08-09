#pragma once

#include <rund/compute/device/info.hpp>
#include <rund/compute/status.hpp>

#include <memory>

namespace rund::compute::detail {

struct DeviceState;

[[nodiscard]] Result<DeviceInfo>
snapshot_device_info(const std::shared_ptr<DeviceState> &state) noexcept;
[[nodiscard]] Status initialize_device_info(DeviceState &state) noexcept;
[[nodiscard]] std::shared_ptr<const DeviceInfo>
device_info_owner(const std::shared_ptr<DeviceState> &state) noexcept;

} // namespace rund::compute::detail

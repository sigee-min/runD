#pragma once

#include <accel/device.hpp>

namespace node_accel_contract::stencil {

[[nodiscard]] bool StorageContract();
[[nodiscard]] bool RunBackend(const rund::AccelDevice &);
[[nodiscard]] bool RunRequiredMetal();
[[nodiscard]] bool RunRequiredVulkan();

} // namespace node_accel_contract::stencil

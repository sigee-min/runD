#pragma once

#include "../capability.hpp"
#include "../request.hpp"
#include "proof.hpp"

namespace rund::node::accel::detail {

[[nodiscard]] bool
device_vsm_capable(const DeviceVsmCapability &capability) noexcept;

[[nodiscard]] bool
device_vsm_request_valid(const DeviceVsmCapability &capability,
                         const DeviceVsmRequest &request) noexcept;

[[nodiscard]] bool
device_vsm_final_valid(const DeviceVsmRequest &request,
                       const DeviceVsmFinal &final) noexcept;

} // namespace rund::node::accel::detail

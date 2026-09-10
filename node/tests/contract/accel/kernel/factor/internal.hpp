#pragma once

#include <accel/device.hpp>

namespace node_accel_contract {

[[nodiscard]] bool BackendRunsFactorDense32(const rund::AccelDevice &pick);
[[nodiscard]] bool BackendRunsFactorDense64(const rund::AccelDevice &pick);

} // namespace node_accel_contract

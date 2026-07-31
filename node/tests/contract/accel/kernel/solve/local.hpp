#pragma once

#include <accel/device.hpp>

namespace node_accel_contract {

[[nodiscard]] bool BackendRunsSolve(const rund::AccelDevice &pick);
[[nodiscard]] bool BackendRunsFactorReuseSolve(const rund::AccelDevice &pick);
[[nodiscard]] bool AvailableBackendsRunSolveNatively();

} // namespace node_accel_contract

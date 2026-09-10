#pragma once

#include <accel/device.hpp>

namespace node_accel_contract {

[[nodiscard]] bool BackendRunsSpectrum(const rund::AccelDevice &pick);
[[nodiscard]] bool SpectrumRejectsInvalidShape(const rund::AccelDevice &pick);
[[nodiscard]] bool AvailableBackendsRunSpectrumNatively();

} // namespace node_accel_contract

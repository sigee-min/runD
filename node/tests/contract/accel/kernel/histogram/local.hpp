#pragma once

#include <accel/device.hpp>

#include "../primitive/local.hpp"

#include <array>

namespace node_accel_contract::histogram {

[[nodiscard]] bool MatchesU32(const rund::AccelDevice &pick);
[[nodiscard]] bool MatchesParallelU32(const rund::AccelDevice &pick);
[[nodiscard]] bool RejectsOutOfRangeBin(const rund::AccelDevice &pick);

} // namespace node_accel_contract::histogram

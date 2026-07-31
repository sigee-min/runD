#pragma once

#include <accel/device.hpp>

#include "../primitive/local.hpp"

namespace node_accel_contract::scatter {

[[nodiscard]] bool MatchesU32(const rund::AccelDevice &pick);
[[nodiscard]] bool MatchesU64(const rund::AccelDevice &pick);
[[nodiscard]] bool RejectsDuplicateIndex(const rund::AccelDevice &pick);
[[nodiscard]] bool ScatterReduceFailuresAreAtomic(
    const rund::AccelDevice &pick);
[[nodiscard]] bool ScatterReduceParallelModes(const rund::AccelDevice &pick);

} // namespace node_accel_contract::scatter

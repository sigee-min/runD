#pragma once

#include <accel/device.hpp>

#include "../primitive/local.hpp"

namespace node_accel_contract::stencil {

[[nodiscard]] bool MatchesU32(const rund::AccelDevice &pick);
[[nodiscard]] bool MatchesSumI32(const rund::AccelDevice &pick);
[[nodiscard]] bool MatchesU64(const rund::AccelDevice &pick);
[[nodiscard]] bool MatchesWideWindowU32(const rund::AccelDevice &pick);
[[nodiscard]] bool MatchesMinU32(const rund::AccelDevice &pick);
[[nodiscard]] bool MatchesMinI32(const rund::AccelDevice &pick);
[[nodiscard]] bool MatchesMaxU64(const rund::AccelDevice &pick);

} // namespace node_accel_contract::stencil

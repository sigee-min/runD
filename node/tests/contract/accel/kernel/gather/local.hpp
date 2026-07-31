#pragma once

#include <accel/device.hpp>

#include "../primitive/local.hpp"

namespace node_accel_contract::gather {

[[nodiscard]] bool MatchesU32(const rund::AccelDevice &pick);
[[nodiscard]] bool MatchesU64(const rund::AccelDevice &pick);
[[nodiscard]] bool PreparedRetainsStorage(const rund::AccelDevice &pick);
[[nodiscard]] bool RejectsOutOfRangeIndex(const rund::AccelDevice &pick);
[[nodiscard]] bool
RejectsBoundedCountOverflowWithoutMutation(const rund::AccelDevice &pick);

} // namespace node_accel_contract::gather

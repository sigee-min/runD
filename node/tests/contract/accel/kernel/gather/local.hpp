#pragma once

#include <accel/device.hpp>
#include <cstddef>

#include "../primitive/local.hpp"

namespace node_accel_contract::gather {

[[nodiscard]] bool MatchesU32(const rund::AccelDevice &pick);
[[nodiscard]] bool MatchesU64(const rund::AccelDevice &pick);
[[nodiscard]] bool PreparedRetainsStorage(const rund::AccelDevice &pick);
[[nodiscard]] bool RejectsOutOfRangeIndex(const rund::AccelDevice &pick,
                                          std::size_t count = 2u);
[[nodiscard]] bool
RejectsBoundedCountOverflowWithoutMutation(const rund::AccelDevice &pick,
                                           std::size_t count = 2u);

} // namespace node_accel_contract::gather

#pragma once

#include <accel/device.hpp>

#include "../primitive/local.hpp"

namespace node_accel_contract::reduce {

[[nodiscard]] bool MatchesU32(const rund::AccelDevice &pick);
[[nodiscard]] bool MatchesU64(const rund::AccelDevice &pick);
[[nodiscard]] bool CountsNonzeroU32(const rund::AccelDevice &pick);
[[nodiscard]] bool CountsNonzeroU64(const rund::AccelDevice &pick);
[[nodiscard]] bool MatchesWideHierarchyU32(const rund::AccelDevice &pick);
[[nodiscard]] bool
MatchesWideHierarchyI64Cancellation(const rund::AccelDevice &pick);
[[nodiscard]] bool RejectsWideU32Overflow(const rund::AccelDevice &pick);
[[nodiscard]] bool RejectsWideU64Overflow(const rund::AccelDevice &pick);
[[nodiscard]] bool MatchesMinU32(const rund::AccelDevice &pick);
[[nodiscard]] bool
MatchesMinU32NonPowerOfTwoBlock(const rund::AccelDevice &pick);
[[nodiscard]] bool MatchesMinI32(const rund::AccelDevice &pick);
[[nodiscard]] bool MatchesMaxU64(const rund::AccelDevice &pick);
[[nodiscard]] bool
MatchesMaxU64NonPowerOfTwoBlock(const rund::AccelDevice &pick);
[[nodiscard]] bool RejectsU32Overflow(const rund::AccelDevice &pick);

} // namespace node_accel_contract::reduce

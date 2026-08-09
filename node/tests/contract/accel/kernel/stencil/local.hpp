#pragma once

#include <accel/device.hpp>

#include "../primitive/local.hpp"
namespace node_accel_contract::stencil {

[[nodiscard]] bool MatchesU32(const rund::AccelDevice &pick);
[[nodiscard]] bool MatchesSumI32(const rund::AccelDevice &pick);
[[nodiscard]] bool MatchesU64(const rund::AccelDevice &pick);
[[nodiscard]] bool MatchesWideWindowU32(const rund::AccelDevice &pick);
[[nodiscard]] bool MatchesCount65U32(const rund::AccelDevice &pick);
[[nodiscard]] bool MatchesRadius64BoundaryU32(const rund::AccelDevice &pick);
[[nodiscard]] bool
MatchesCapabilitySharedBoundaryU32(const rund::AccelDevice &pick,
                                   rund::kernel::u64 radius);
[[nodiscard]] bool MatchesPrefixDifferenceU32(const rund::AccelDevice &pick);
[[nodiscard]] bool
MatchesForcedPrefixDifferenceU32(const rund::AccelDevice &pick);
[[nodiscard]] bool
MatchesForcedPrefixDifferenceU64(const rund::AccelDevice &pick);
[[nodiscard]] bool MatchesDeepPrefixHierarchyU32(const rund::AccelDevice &pick);
[[nodiscard]] bool MatchesMinU32(const rund::AccelDevice &pick);
[[nodiscard]] bool MatchesMinI32(const rund::AccelDevice &pick);
[[nodiscard]] bool MatchesMaxU64(const rund::AccelDevice &pick);
[[nodiscard]] bool
MatchesBlockPrefixSuffixMinI32(const rund::AccelDevice &pick);
[[nodiscard]] bool
MatchesBlockPrefixSuffixMaxI32(const rund::AccelDevice &pick);
[[nodiscard]] bool
MatchesBlockPrefixSuffixMinU64(const rund::AccelDevice &pick);
[[nodiscard]] bool
MatchesBlockPrefixSuffixMaxU64(const rund::AccelDevice &pick);
[[nodiscard]] bool
MatchesForcedBlockPrefixSuffixMinI32(const rund::AccelDevice &pick);
[[nodiscard]] bool
MatchesForcedBlockPrefixSuffixMaxI32(const rund::AccelDevice &pick);
[[nodiscard]] bool
MatchesForcedBlockPrefixSuffixMinU64(const rund::AccelDevice &pick);
[[nodiscard]] bool
MatchesForcedBlockPrefixSuffixMaxU64(const rund::AccelDevice &pick);

} // namespace node_accel_contract::stencil

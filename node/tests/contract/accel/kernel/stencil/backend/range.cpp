#include "local.hpp"

namespace node_accel_contract::stencil::backend {

bool RunRange(const rund::AccelDevice &pick) {
  return StencilMatch(stencil::MatchesTiledDifference(pick),
                      "sum.u32-u64.tiled") &&
         StencilMatch(stencil::MatchesForcedPrefixDifferenceU32(pick),
                      "sum.u32.direct-prefix-difference") &&
         StencilMatch(stencil::MatchesForcedPrefixDifferenceU64(pick),
                      "sum.u64.direct-prefix-difference") &&
         StencilMatch(stencil::MatchesDeepPrefixHierarchy(pick),
                      "sum.u32-u64.prefix-hierarchy") &&
         StencilMatch(stencil::MatchesForcedBlockPrefixSuffixMinI32(pick),
                      "min.i32.direct-block-prefix-suffix") &&
         StencilMatch(stencil::MatchesForcedBlockPrefixSuffixMaxI32(pick),
                      "max.i32.direct-block-prefix-suffix") &&
         StencilMatch(stencil::MatchesForcedBlockPrefixSuffixMinU64(pick),
                      "min.u64.direct-block-prefix-suffix") &&
         StencilMatch(stencil::MatchesForcedBlockPrefixSuffixMaxU64(pick),
                      "max.u64.direct-block-prefix-suffix");
}

} // namespace node_accel_contract::stencil::backend

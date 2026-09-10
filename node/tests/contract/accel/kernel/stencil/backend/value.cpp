#include "local.hpp"

namespace node_accel_contract::stencil::backend {

bool RunValue(const rund::AccelDevice &pick) {
  return StencilMatch(stencil::MatchesU64(pick), "sum.u64") &&
         StencilMatch(stencil::MatchesMinU32(pick), "min.u32") &&
         StencilMatch(stencil::MatchesMinI32(pick), "min.i32") &&
         StencilMatch(stencil::MatchesMaxU64(pick), "max.u64") &&
         StencilMatch(stencil::MatchesBlockPrefixSuffixMinI32(pick),
                      "min.i32.block-prefix-suffix") &&
         StencilMatch(stencil::MatchesBlockPrefixSuffixMaxI32(pick),
                      "max.i32.block-prefix-suffix") &&
         StencilMatch(stencil::MatchesBlockPrefixSuffixMinU64(pick),
                      "min.u64.block-prefix-suffix") &&
         StencilMatch(stencil::MatchesBlockPrefixSuffixMaxU64(pick),
                      "max.u64.block-prefix-suffix");
}

} // namespace node_accel_contract::stencil::backend

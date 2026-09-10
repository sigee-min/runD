#include "local.hpp"

namespace node_accel_contract::stencil::backend {

bool RunGeometry(const rund::AccelDevice &pick) {
  return StencilMatch(stencil::MatchesWideWindowU32(pick), "sum.u32.radius2") &&
         StencilMatch(stencil::MatchesCount65U32(pick), "sum.u32.count65") &&
         StencilMatch(stencil::MatchesRadius64BoundaryU32(pick),
                      "sum.u32.count259.radius64") &&
         StencilMatch(stencil::MatchesPrefixDifferenceU32(pick),
                      "sum.u32.prefix-difference");
}

} // namespace node_accel_contract::stencil::backend

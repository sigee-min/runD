#include "../contract.hpp"
#include "local.hpp"

namespace node_accel_contract::stencil {

bool RunBackend(const rund::AccelDevice &pick) {
  return backend::StencilMatch(BindingsContract(), "bindings") &&
         backend::StencilMatch(MatchesU32(pick), "sum.u32") &&
         backend::RunGeometry(pick) && backend::RunValue(pick);
}

bool RunRequiredMetal() { return backend::RunMetalRequired(); }

bool RunRequiredVulkan() { return backend::RunVulkanRequired(); }

} // namespace node_accel_contract::stencil

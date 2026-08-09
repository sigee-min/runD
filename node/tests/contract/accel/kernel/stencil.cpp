#include "stencil/contract.hpp"

namespace node_accel_contract {

bool BackendRunsStencil(const rund::AccelDevice &pick) {
  return stencil::RunBackend(pick);
}

bool RequiredMetalRunsStencil() { return stencil::RunRequiredMetal(); }

bool RequiredVulkanRunsStencil() { return stencil::RunRequiredVulkan(); }

} // namespace node_accel_contract

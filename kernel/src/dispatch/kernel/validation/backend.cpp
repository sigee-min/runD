#include "../local.hpp"

namespace rund::kernel::dispatch::detail {

const char* ValidateCommonBackendCapabilities(const WorkerBackendCapabilities& capabilities) {
  if (!capabilities.width_matches_request) {
    return "pool_width_mismatch";
  }
  if (capabilities.is_nested) {
    return "pool_nested_dispatch";
  }
  if (capabilities.affinity_policy != WorkerAffinityPolicy::Static) {
    return "pool_policy_mismatch";
  }
  return nullptr;
}

} // namespace rund::kernel::dispatch::detail

#include "local.hpp"

namespace rund::node::accel::detail {

FinalGraphSteps
RejectFinalGraphSteps(const char *const reason,
                      const rund::kernel::FusionPlan &fusion) noexcept {
  return FinalGraphSteps{
      .fused_operation_count = fusion.original_node_count,
      .fusion_rejection_count = fusion.rejected_edge_count,
      .fusion_reason = fusion.reason,
      .ok = false,
      .reason = reason,
  };
}

} // namespace rund::node::accel::detail

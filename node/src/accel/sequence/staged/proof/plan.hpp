#pragma once

#include "bulk.hpp"

namespace rund::node::accel::detail {

[[nodiscard]] inline StagedProof PlanStagedWindow(
    const rund::kernel::BindingSet& bindings,
    const rund::kernel::ComputeDispatchWindow& window) noexcept {
  if (!StagedSequenceRangeOk(bindings, window) ||
      bindings.staged_output_stride < bindings.output_bytes_per_tile) {
    return {};
  }
  bool identity = true;
  if (!StagedIdentityAndOutputsOk(bindings, window, identity)) {
    return {};
  }
  return StagedProof{
      true, StagedBulkAllowed(bindings, window, identity), bindings, window};
}

}  // namespace rund::node::accel::detail

#include "local.hpp"

#include <algorithm>

namespace rund::kernel::reduction::fold {

FoldGraphValidationResult ValidateFoldGraphSlots(const FoldGraphView graph,
                                                 const std::span<u64> defined_markers,
                                                 const u32 required_slots) {
  if (required_slots == 0u) {
    return FailFoldGraphValidation("fold_graph_scratch_capacity_exceeded");
  }
  if (defined_markers.size() < static_cast<std::size_t>(required_slots)) {
    return FailFoldGraphValidation("fold_graph_scratch_capacity_exceeded", required_slots);
  }
  std::fill(defined_markers.begin(), defined_markers.begin() + required_slots, 0u);
  for (u32 partition = 0u; partition < graph.partition_count; ++partition) {
    const u32 slot = graph.partition_fold_slots[partition];
    if (slot >= required_slots || defined_markers[slot] != 0u) {
      return FailFoldGraphValidation("fold_graph_slot_out_of_range", required_slots);
    }
    defined_markers[slot] = kFoldGraphValidationPartitionMarker;
  }
  return FoldGraphValidationResult{.ok = true, .reason = "pass"};
}

} // namespace rund::kernel::reduction::fold

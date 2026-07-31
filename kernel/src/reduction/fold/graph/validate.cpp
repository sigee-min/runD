#include "validation/local.hpp"

#include <kernel/reduction/fold/graph/api.hpp>

namespace rund::kernel::reduction::fold {

FoldGraphValidationResult ValidateFoldGraphWithMarkers(const FoldGraphView graph,
                                                       const std::span<u64> defined_markers) {
  FoldGraphValidationResult validation = ValidateFoldGraphShape(graph);
  if (!validation.ok) {
    return validation;
  }
  const u32 required_slots = FoldGraphScratchSlotCount(graph);
  validation = ValidateFoldGraphSlots(graph, defined_markers, required_slots);
  if (!validation.ok) {
    return validation;
  }
  const FoldPrimitiveSpec primitive = DescribeFoldOperation(graph.operation);
  u32 final_slot = 0u;
  validation = ValidateFoldGraphEdges(graph, defined_markers, primitive, required_slots, final_slot);
  if (!validation.ok) {
    return validation;
  }
  if (graph.final_slot != final_slot) {
    return FailFoldGraphValidation("fold_graph_final_slot_mismatch", required_slots);
  }
  validation = ValidateFoldGraphNodes(graph, defined_markers, primitive, required_slots);
  if (!validation.ok) {
    return validation;
  }
  return FoldGraphValidationResult{
      .ok = true,
      .reason = "pass",
      .scratch_slot_count = required_slots,
      .final_slot = final_slot,
      .dag_validated = true,
      .slot_bounds_validated = true,
      .padding_law_validated = true,
  };
}

} // namespace rund::kernel::reduction::fold

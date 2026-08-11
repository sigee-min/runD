#include "local.hpp"

#include "src/compute/pipeline/residency/footprint.hpp"
#include "src/compute/pipeline/residency/planner.hpp"

namespace rund_node_test_pipeline_residency {

int CheckFootprints() {
  using namespace rund::compute::detail::residency;
  DemandEpoch window{.node = 4u, .tile = 3u};
  if (!ProjectWindow(
          WindowFootprint{
              .input_resource = 1u,
              .output_resource = 2u,
              .input_elements = 64u,
              .output_first = 16u,
              .output_count = 16u,
              .window_size = 5u,
              .stride = 1u,
              .pad_left = 2u,
              .element_bytes = 4u,
          },
          32u, window)) {
    return 1;
  }
  // Output [16,31] needs the [14,33] halo: input pages 1..4 and output
  // pages 2..3. Canonicalization belongs to the planner, so projection keeps
  // the two resource domains explicit.
  auto window_plan = PlanResidency(GraphPlanInput{
      .page_bytes = 32u, .frame_capacity = 6u, .epochs = {window}});
  if (!window_plan || window_plan.plan.uses().size() != 6u) {
    return 2;
  }

  std::vector<DemandEpoch> scan;
  if (!ProjectScan(
          ScanFootprint{
              .input_resource = 1u,
              .output_resource = 2u,
              .partial_resource = 3u,
              .element_count = 32u,
              .tile_elements = 16u,
              .element_bytes = 4u,
          },
          64u, scan) ||
      scan.size() != 5u || scan[0].node != 0u || scan[2].node != 1u ||
      scan[3].node != 2u) {
    return 3;
  }
  auto scan_plan = PlanResidency(
      GraphPlanInput{.page_bytes = 64u, .frame_capacity = 3u, .epochs = scan});
  return scan_plan && scan_plan.plan.epochs().size() == 5u ? 0 : 4;
}

} // namespace rund_node_test_pipeline_residency

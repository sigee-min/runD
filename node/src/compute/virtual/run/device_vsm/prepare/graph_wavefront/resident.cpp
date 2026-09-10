#include "internal.hpp"

namespace rund::compute::detail::device_vsm_product_detail {

bool project_graph_resident(
    const residency::TiledGraphPlan &plan, const residency::Pool &pool,
    const std::span<const std::uint32_t> graph_input_resources,
    const node::accel::detail::DeviceVsmGraphWavefrontProof &wavefront,
    node::accel::detail::DeviceVsmGraphResidentProof &result,
    const char *&reason) noexcept {
  result = {};
  reason = nullptr;
  GraphResidentDraft draft{plan,
                           pool,
                           graph_input_resources,
                           wavefront,
                           result,
                           reason,
                           plan.resources(),
                           plan.stages(),
                           plan.physical_classes()};
  if (!validate_graph_resident_shape(draft) ||
      !project_graph_resident_resources(draft)) {
    return false;
  }
  if (!project_graph_page_map(plan, graph_input_resources, result.page_map)) {
    remember_graph_resident_reason(GraphResidentShapeInvalid, reason);
    return false;
  }
  if (!project_graph_resident_owners(draft) ||
      !project_graph_resident_stages(draft)) {
    return false;
  }
  return true;
}

} // namespace rund::compute::detail::device_vsm_product_detail

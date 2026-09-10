#include "internal.hpp"

#include <utility>

namespace rund::node::accel::detail {

#if defined(__APPLE__) && defined(RUND_NODE_HAVE_METAL_SDK)

rund::AccelCheck MetalPipelineBuild::Admit() {
  bool compact_input = false;
  const rund::AccelCheck valid =
      metal_pipeline_admit_internal::ValidateMetalPipelineInput(*this,
                                                                compact_input);
  if (!valid.ok) {
    return valid;
  }
  const rund::AccelCheck aggregate =
      metal_pipeline_admit_internal::AdmitMetalAggregate(*this, compact_input);
  if (!aggregate.ok) {
    return aggregate;
  }
  if (aggregate_selected) {
    return aggregate;
  }
  if (compact_input) {
    // The compact representation intentionally owns no canonical occurrence
    // stream. Reaching this point means the common proof and native admission
    // disagreed, so fail closed instead of manufacturing a second authority.
    return rund::AccelCheck{false, "accel_kernel_run_invalid"};
  }
  MetalSpatialWindowProof spatial_window{};
  std::uint32_t state_count = 0u;
  const rund::AccelCheck resources =
      metal_pipeline_admit_internal::PrepareMetalPipelineResources(
          *this, spatial_window, state_count);
  if (!resources.ok) {
    return resources;
  }
  return metal_pipeline_admit_internal::PrepareMetalPipelineRoutes(
      *this, std::move(spatial_window), state_count);
}

#endif

} // namespace rund::node::accel::detail

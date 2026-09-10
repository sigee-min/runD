#pragma once

#include "../../build.hpp"
#include "../spatial_window/proof.hpp"

namespace rund::node::accel::detail {

#if defined(__APPLE__) && defined(RUND_NODE_HAVE_METAL_SDK)
namespace metal_pipeline_admit_internal {

[[nodiscard]] rund::AccelCheck
ValidateMetalPipelineInput(MetalPipelineBuild &build, bool &compact_input);

[[nodiscard]] rund::AccelCheck AdmitMetalAggregate(MetalPipelineBuild &build,
                                                   bool compact_input);

[[nodiscard]] rund::AccelCheck
PrepareMetalPipelineResources(MetalPipelineBuild &build,
                              MetalSpatialWindowProof &spatial_window,
                              std::uint32_t &state_count);

[[nodiscard]] rund::AccelCheck
PrepareMetalPipelineRoutes(MetalPipelineBuild &build,
                           MetalSpatialWindowProof &&spatial_window,
                           std::uint32_t state_count);

[[nodiscard]] rund::AccelCheck PrepareMetalRecurrenceRoute(
    PreparedKernelTemplateRegistry &registry, MetalAdapter &adapter,
    const rund::AccelDevice &pick, const BackendRun &signature,
    const MapRecurrence &recurrence, const BoundControl &control,
    std::shared_ptr<void> &resources, std::uint32_t iterations);

} // namespace metal_pipeline_admit_internal
#endif

} // namespace rund::node::accel::detail

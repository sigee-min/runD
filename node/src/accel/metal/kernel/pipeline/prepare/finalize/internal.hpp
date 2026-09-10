#pragma once

#include "../../build.hpp"
#include "../../icb.hpp"

namespace rund::node::accel::detail {

#if defined(__APPLE__) && defined(RUND_NODE_HAVE_METAL_SDK)

// Finalization passes data between the cold projection phases only.  This is
// deliberately not a second Pipeline owner: all retained state stays in the
// MetalPipelineBuild/Pipeline objects, while these values describe one
// immutable capture result and its transient accounting.
struct MetalPipelineFinalizeProjection final {
  MetalIcbChunkPlan actual_icb_plan{};
  std::uint64_t icb_device_bytes{};
  std::uint64_t identity_index_bytes{};
  std::size_t native_window_capacity{};
  std::size_t parameter_residency_index{};
  bool uses_parameters{};
  bool direct_window{};
  bool spatial_window{};
};

[[nodiscard]] rund::AccelCheck FinalizeMetalCapture(MetalPipelineBuild &build);
[[nodiscard]] rund::AccelCheck
FinalizeMetalProjection(MetalPipelineBuild &build,
                        MetalPipelineFinalizeProjection &projection);
[[nodiscard]] rund::AccelCheck
FinalizeMetalProjectionIdentity(MetalPipelineBuild &build,
                                MetalPipelineFinalizeProjection &projection);
[[nodiscard]] rund::AccelCheck
FinalizeMetalProjectionWindows(MetalPipelineBuild &build,
                               MetalPipelineFinalizeProjection &projection);
[[nodiscard]] rund::AccelCheck
FinalizeMetalProjectionRanges(MetalPipelineBuild &build,
                              MetalPipelineFinalizeProjection &projection);
[[nodiscard]] rund::AccelCheck
FinalizeMetalNative(MetalPipelineBuild &build,
                    MetalPipelineFinalizeProjection &projection);
[[nodiscard]] rund::AccelCheck
FinalizeMetalOwner(MetalPipelineBuild &build,
                   const MetalPipelineFinalizeProjection &projection,
                   std::shared_ptr<void> &prepared,
                   PreparedPipelineMemory &memory);

#endif

} // namespace rund::node::accel::detail

#pragma once

#include "../../build.hpp"

#include <cstddef>
#include <cstdint>

namespace rund::node::accel::detail::metal_pipeline_describe_internal {

#if defined(__APPLE__) && defined(RUND_NODE_HAVE_METAL_SDK)

// These values are one-call derived facts shared by Describe's direct owners.
// They do not become fields on MetalPipelineBuild or a second retained state.
struct MetalPipelineDescribeCapacity final {
  std::size_t template_step_capacity{};
  std::size_t status_source_capacity{};
  std::size_t telemetry_capacity{};
  std::uint32_t status_entry_capacity{};
};

[[nodiscard]] rund::AccelCheck
DescribeMetalAggregate(MetalPipelineBuild &build);

[[nodiscard]] rund::AccelCheck
DescribeMetalCapacity(MetalPipelineBuild &build,
                      MetalPipelineDescribeCapacity &capacity);

[[nodiscard]] rund::AccelCheck
DescribeMetalTemplates(MetalPipelineBuild &build,
                       const MetalPipelineDescribeCapacity &capacity);

[[nodiscard]] rund::AccelCheck
DescribeMetalOccurrences(MetalPipelineBuild &build);

[[nodiscard]] rund::AccelCheck DescribeMetalProfile(MetalPipelineBuild &build);

[[nodiscard]] rund::AccelCheck
DescribeMetalPacking(MetalPipelineBuild &build,
                     const MetalPipelineDescribeCapacity &capacity);

#endif

} // namespace rund::node::accel::detail::metal_pipeline_describe_internal

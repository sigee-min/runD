#pragma once

#include "../local.hpp"

namespace rund::node::accel::detail {

[[nodiscard]] bool PrepareMetalRangeTemps(MetalAdapter &adapter,
                                          MetalRangeResources &resources);
[[nodiscard]] bool
PrepareMetalRangeControl(const rund::AccelDevice &pick,
                         const BoundControl *bound,
                         MetalRangeResources &resources,
                         const MetalKernelImmutablePipelines *pipelines);

} // namespace rund::node::accel::detail

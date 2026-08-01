#pragma once

#include "model.hpp"

#include <accel/check.hpp>

namespace rund::node::accel::detail {

#if defined(__APPLE__) && defined(RUND_NODE_HAVE_METAL_SDK)

[[nodiscard]] rund::AccelCheck
PrepareMetalNestedAggregate(MetalAdapter &adapter,
                            MetalNestedAggregate &aggregate);

[[nodiscard]] rund::AccelCheck
EncodeMetalNestedAggregate(const MetalNestedAggregate &aggregate,
                           id<MTLBuffer> control, id<MTLBuffer> step_control,
                           RUNDMetalPipelineCapture *encoder);

#endif

} // namespace rund::node::accel::detail

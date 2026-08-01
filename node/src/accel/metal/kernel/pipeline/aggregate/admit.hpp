#pragma once

#include "model.hpp"

#include <accel/check.hpp>

#include "../../../../kernel/recurrence.hpp"

namespace rund::node::accel::detail {

#if defined(__APPLE__) && defined(RUND_NODE_HAVE_METAL_SDK)

[[nodiscard]] rund::AccelCheck AdmitMetalNestedAggregate(
    const NestedAggregate &source, const PreparedPipelineStatusLayout &status,
    bool profile_steps, MetalKernelContext &context, MetalNestedAggregate &out);

#endif

} // namespace rund::node::accel::detail

#pragma once

#include <accel/check.hpp>

#include <kernel/program/compute/graph/schema.hpp>

#include <memory>

namespace rund::node::accel::detail {

#if defined(__APPLE__) && defined(RUND_NODE_HAVE_METAL_SDK)
struct MetalKernelImmutablePipelines;

[[nodiscard]] rund::AccelCheck FreezeMetalPrimitivePipelines(
    rund::kernel::NodeKind kind, const std::shared_ptr<void> &resource,
    std::shared_ptr<const MetalKernelImmutablePipelines> &out);
#endif

} // namespace rund::node::accel::detail

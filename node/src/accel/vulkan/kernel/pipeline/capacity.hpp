#pragma once

#include <kernel/program/compute/limit.hpp>

#include <cstdint>

namespace rund::node::accel::detail {

// Native Vulkan template materialization is deliberately bounded by one
// canonical ComputeIR envelope. This is a backend compile/object ceiling, not
// the compact Pipeline route/status capacity. Public plan-only inspection may
// exceed it; prepare rejects before registry publication or any Vulkan call.
inline constexpr std::uint64_t kVulkanPipelineTemplateStepCapacity =
    rund::kernel::kMaxComputeNodeCount;

} // namespace rund::node::accel::detail

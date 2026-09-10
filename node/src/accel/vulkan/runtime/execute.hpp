#pragma once

#include "../adapter/state.hpp"

#include <kernel/program/compute/artifact.hpp>
#include <kernel/program/compute/backend.hpp>
#include <kernel/program/compute/binding/model.hpp>
#include <kernel/program/compute/model.hpp>

namespace rund::node::accel::detail {

#if defined(RUND_NODE_HAVE_VULKAN_SDK)

[[nodiscard]] bool ExecuteVulkan(
    void *context, const rund::kernel::ComputePlan &plan,
    const rund::kernel::LoweringArtifact &artifact,
    const rund::kernel::ComputeDispatchWindow *windows,
    rund::kernel::u64 window_count,
    const rund::kernel::BindingSet &bindings);

#endif // defined(RUND_NODE_HAVE_VULKAN_SDK)

} // namespace rund::node::accel::detail

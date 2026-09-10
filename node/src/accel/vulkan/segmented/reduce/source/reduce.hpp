#pragma once

#include "../model.hpp"

#include "../../../../kernel/backend/source/sink.hpp"

namespace rund::node::accel::detail {

#if defined(RUND_NODE_HAVE_VULKAN_SDK)

[[nodiscard]] bool EmitVulkanSegmentedReduceReduceSource(
    backend_source_recipe::CountSink &sink,
    const rund::kernel::SegmentedReducePlan &plan,
    rund::kernel::ComputeDomain domain) noexcept;
[[nodiscard]] bool EmitVulkanSegmentedReduceReduceSource(
    backend_source_recipe::StringSink &sink,
    const rund::kernel::SegmentedReducePlan &plan,
    rund::kernel::ComputeDomain domain);

#endif

} // namespace rund::node::accel::detail

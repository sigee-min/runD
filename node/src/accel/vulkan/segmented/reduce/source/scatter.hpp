#pragma once

#include "../model.hpp"

#include "../../../../kernel/backend/source/sink.hpp"

namespace rund::node::accel::detail {

#if defined(RUND_NODE_HAVE_VULKAN_SDK)

[[nodiscard]] bool EmitVulkanSegmentedReduceScatterSource(
    backend_source_recipe::CountSink &sink) noexcept;
[[nodiscard]] bool
EmitVulkanSegmentedReduceScatterSource(backend_source_recipe::StringSink &sink);

#endif

} // namespace rund::node::accel::detail

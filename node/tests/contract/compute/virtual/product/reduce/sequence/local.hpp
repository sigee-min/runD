#pragma once

#include "../local.hpp"

#if defined(RUND_NODE_HAVE_VULKAN_SDK)

namespace rund_node_test_virtual::product::reduce::sequence_detail {

void PrintTiledGraphEvidence(
    rund::compute::Backend backend,
    const std::shared_ptr<rund::compute::detail::VirtualPipelineState> &state,
    const ProductRouteObservation &observation);

[[nodiscard]] bool InspectGraphSequence(
    const std::shared_ptr<rund::compute::detail::VirtualPipelineState> &state,
    bool after_run) noexcept;

} // namespace rund_node_test_virtual::product::reduce::sequence_detail

#endif

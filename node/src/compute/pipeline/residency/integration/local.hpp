#pragma once

#include "../integration.hpp"

namespace rund::compute::detail {

[[nodiscard]] bool graph_frame_region(const residency::Pool &pool,
                                      std::uint32_t bank,
                                      std::uint32_t physical_id,
                                      residency::FrameRegion &region) noexcept;

[[nodiscard]] Status
bind_graph_semantic_residency(const PipelineMemoryPlan &plan,
                              PipelineState &state) noexcept;

[[nodiscard]] Status bind_graph_residency(const PipelineMemoryPlan &plan,
                                          PipelineState &state) noexcept;

[[nodiscard]] Status bind_direct_residency(const PipelineMemoryPlan &plan,
                                           PipelineState &state) noexcept;

} // namespace rund::compute::detail

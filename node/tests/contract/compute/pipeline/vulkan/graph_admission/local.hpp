#pragma once

#include <memory>

namespace rund::compute::detail {
struct PipelineState;
}

namespace node_compute_pipeline_vulkan_graph_admission {

[[nodiscard]] bool PreparedPipelineReleaseUnlocks(
    const std::shared_ptr<rund::compute::detail::PipelineState> &state);
[[nodiscard]] bool RejectRecordedGraphReuse(
    const std::shared_ptr<rund::compute::detail::PipelineState> &state);
[[nodiscard]] bool RejectRecordedGraphReprepare(
    const std::shared_ptr<rund::compute::detail::PipelineState> &state);

[[nodiscard]] bool CheckGenerationSeedDomains(
    const std::shared_ptr<rund::compute::detail::PipelineState> &state);

} // namespace node_compute_pipeline_vulkan_graph_admission

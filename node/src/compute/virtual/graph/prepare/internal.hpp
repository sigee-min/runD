#pragma once

#include "../../../backend.hpp"
#include "../../../device/residency/pool.hpp"
#include "../../../graph/compile/slice.hpp"
#include "../../../pipeline/local.hpp"
#include "../../../pipeline/residency/integration.hpp"
#include "../../../pipeline/residency/planner.hpp"
#include "../../host_ring.hpp"
#include "../../local.hpp"
#include "../../stats.hpp"

#include <cstdint>
#include <memory>
#include <span>
#include <vector>

namespace rund::compute::detail::virtual_graph_prepare_detail {

// This is an invocation-local value record. It owns no physical authority:
// Pool, Pipeline, and the immutable ResidencyPlan remain owned by their
// existing subsystems. The record only carries phase results until the final
// assembly publishes one VirtualPipelineState.
struct GraphPreparationDraft final {
  std::shared_ptr<ProgramState> program;
  std::span<const std::shared_ptr<VirtualBufferState>> inputs{};
  std::shared_ptr<VirtualBufferState> output;
  ResidencyConfig config{};
  GraphPageMap page_map{};
  VirtualGeometry geometry{};
  const VirtualBufferState *input{};
  bool graph_reduction{};
  bool graph_pointwise{};

  graph_compile::TiledGraphSlices sliced{};
  graph_compile::TiledGraphSliceResource terminal_input{};
  std::uint64_t page_count{};
  std::uint64_t prefetch_distance{};
  std::uint64_t input_page_bytes{};
  std::uint64_t intermediate_page_bytes{};
  std::uint64_t control_page_bytes{};
  std::uint64_t output_page_bytes{};
  std::uint64_t host_input_page_bytes{};
  std::uint64_t host_frame_bytes{};
  std::uint64_t logical_bytes{};

  std::vector<residency::TiledGraphResourceInput> graph_resources;
  std::vector<residency::TiledGraphStageInput> graph_stages;
  residency::PlanResult topology{};
  residency::PlanResult planned{};
  std::uint64_t frames{};
  std::uint64_t host_frames{};
  std::uint64_t host_output_frames{};
  std::uint64_t resident_bytes{};
  std::uint64_t host_storage_bytes{};

  std::shared_ptr<const residency::ResidencyPlan> pages;
  std::shared_ptr<residency::Pool> pool;
  std::vector<std::shared_ptr<PipelineState>> graph_pipelines;
  std::shared_ptr<PipelineState> device_vsm_semantic_pipeline;
};

[[nodiscard]] Status validate_graph_request(GraphPreparationDraft &) noexcept;
[[nodiscard]] bool
valid_graph_map_shape(std::span<const residency::GraphPageRemap> remaps,
                      std::size_t frame_count, std::size_t page_count) noexcept;
[[nodiscard]] Status compile_graph_slices(GraphPreparationDraft &) noexcept;
[[nodiscard]] Status calculate_graph_bytes(GraphPreparationDraft &) noexcept;
[[nodiscard]] Status materialize_graph_inputs(GraphPreparationDraft &) noexcept;
[[nodiscard]] Status plan_graph_topology(GraphPreparationDraft &) noexcept;
[[nodiscard]] Status project_graph_budget(GraphPreparationDraft &) noexcept;
[[nodiscard]] Result<std::shared_ptr<VirtualPipelineState>>
assemble_graph(GraphPreparationDraft &) noexcept;

} // namespace rund::compute::detail::virtual_graph_prepare_detail

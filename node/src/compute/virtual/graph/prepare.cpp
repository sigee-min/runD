#include "prepare/internal.hpp"

namespace rund::compute::detail {
namespace {

using VirtualResult = Result<std::shared_ptr<VirtualPipelineState>>;

[[nodiscard]] VirtualResult fail(const Reason reason) noexcept {
  return VirtualResult::fail(reason);
}

} // namespace

Result<std::shared_ptr<VirtualPipelineState>> prepare_virtual_tiled_graph(
    const std::shared_ptr<ProgramState> &program,
    const std::span<const std::shared_ptr<VirtualBufferState>> inputs,
    const std::shared_ptr<VirtualBufferState> &output,
    const ResidencyConfig config, const GraphPageMap page_map,
    const VirtualGeometry &geometry) noexcept {
  virtual_graph_prepare_detail::GraphPreparationDraft draft{};
  draft.program = program;
  draft.inputs = inputs;
  draft.output = output;
  draft.config = config;
  draft.page_map = page_map;
  draft.geometry = geometry;

  const auto validate =
      virtual_graph_prepare_detail::validate_graph_request(draft);
  if (!validate) {
    return fail(validate.reason());
  }
  const auto slices = virtual_graph_prepare_detail::compile_graph_slices(draft);
  if (!slices) {
    return fail(slices.reason());
  }
  const auto bytes = virtual_graph_prepare_detail::calculate_graph_bytes(draft);
  if (!bytes) {
    return fail(bytes.reason());
  }
  const auto materialized =
      virtual_graph_prepare_detail::materialize_graph_inputs(draft);
  if (!materialized) {
    return fail(materialized.reason());
  }
  const auto topology =
      virtual_graph_prepare_detail::plan_graph_topology(draft);
  if (!topology) {
    return fail(topology.reason());
  }
  const auto budget = virtual_graph_prepare_detail::project_graph_budget(draft);
  if (!budget) {
    return fail(budget.reason());
  }
  return virtual_graph_prepare_detail::assemble_graph(draft);
}

Result<std::shared_ptr<VirtualPipelineState>> prepare_virtual_graph_reduction(
    const std::shared_ptr<ProgramState> &program,
    const std::span<const std::shared_ptr<VirtualBufferState>> inputs,
    const std::shared_ptr<VirtualBufferState> &output,
    const ResidencyConfig config, const GraphPageMap page_map,
    const VirtualGeometry &geometry) noexcept {
  return prepare_virtual_tiled_graph(program, inputs, output, config, page_map,
                                     geometry);
}

Result<std::shared_ptr<VirtualPipelineState>> prepare_virtual_graph_pointwise(
    const std::shared_ptr<ProgramState> &program,
    const std::span<const std::shared_ptr<VirtualBufferState>> inputs,
    const std::shared_ptr<VirtualBufferState> &output,
    const ResidencyConfig config, const GraphPageMap page_map,
    const VirtualGeometry &geometry) noexcept {
  return prepare_virtual_tiled_graph(program, inputs, output, config, page_map,
                                     geometry);
}

} // namespace rund::compute::detail

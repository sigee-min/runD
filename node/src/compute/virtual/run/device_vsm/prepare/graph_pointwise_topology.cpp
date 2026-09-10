#include "graph_pointwise_topology.hpp"

#include <algorithm>
#include <bit>

namespace rund::compute::detail::device_vsm_product_detail {

bool project_graph_pointwise_topology(
    const residency::TiledGraphPlan &plan,
    const std::span<const std::uint32_t> external_inputs,
    node::accel::detail::DeviceVsmGraphPointwiseTopology &result) noexcept {
  namespace accel = node::accel::detail;
  result = {};
  const std::span<const residency::TiledGraphStage> stages = plan.stages();
  if (stages.size() < 2u ||
      stages.size() > accel::DeviceVsmGraphStageCapacity ||
      external_inputs.empty() ||
      external_inputs.size() >= accel::DeviceVsmResidentCapacity) {
    return false;
  }
  result.stage_count = static_cast<std::uint8_t>(stages.size());
  result.external_input_count =
      static_cast<std::uint8_t>(external_inputs.size());
  std::uint8_t referenced_outputs = 0u;
  for (std::size_t stage_index = 0u; stage_index < stages.size();
       ++stage_index) {
    const residency::TiledGraphStage &stage = stages[stage_index];
    accel::DeviceVsmGraphPointwiseStageTopology &projected =
        result.stages[stage_index];
    std::uint8_t input_mask = 0u;
    std::size_t write_count = 0u;
    for (const residency::TiledGraphPort port : stage.ports) {
      const residency::TiledGraphResource *const resource =
          plan.resource(port.resource);
      if (resource == nullptr) {
        return false;
      }
      if (port.access == residency::Access::Read) {
        if (port.program_port >= accel::DeviceVsmGraphStageInputCapacity ||
            (input_mask & (std::uint8_t{1u} << port.program_port)) != 0u) {
          return false;
        }
        accel::DeviceVsmGraphValueSource source{};
        const auto external = std::find(external_inputs.begin(),
                                        external_inputs.end(), port.resource);
        if (external != external_inputs.end()) {
          source.kind = accel::DeviceVsmGraphValueSourceKind::ExternalInput;
          source.index = static_cast<std::uint8_t>(
              std::distance(external_inputs.begin(), external));
        } else if (resource->producer_stage < stage_index &&
                   resource->producer_stage < stages.size()) {
          source.kind = accel::DeviceVsmGraphValueSourceKind::StageOutput;
          source.index = static_cast<std::uint8_t>(resource->producer_stage);
          referenced_outputs = static_cast<std::uint8_t>(
              referenced_outputs | (std::uint8_t{1u} << source.index));
        } else {
          return false;
        }
        projected.inputs[port.program_port] = source;
        input_mask = static_cast<std::uint8_t>(
            input_mask | (std::uint8_t{1u} << port.program_port));
      } else if (port.access == residency::Access::Write) {
        const bool terminal = stage_index + 1u == stages.size();
        const bool owned_output =
            terminal
                ? resource->kind ==
                          residency::GraphResourceKind::ExternalOutput &&
                      resource->producer_stage == residency::NoGraphStage
                : resource->kind == residency::GraphResourceKind::Internal &&
                      resource->producer_stage == stage_index;
        if (++write_count != 1u || !owned_output) {
          return false;
        }
      } else {
        return false;
      }
    }
    projected.input_count = static_cast<std::uint8_t>(
        std::popcount(static_cast<unsigned>(input_mask)));
    const std::uint16_t dense =
        projected.input_count == 0u
            ? 0u
            : static_cast<std::uint16_t>(
                  (std::uint16_t{1u} << projected.input_count) - 1u);
    if (write_count != 1u || projected.input_count == 0u ||
        input_mask != dense) {
      return false;
    }
  }
  const std::uint8_t required = static_cast<std::uint8_t>(
      (std::uint8_t{1u} << (stages.size() - 1u)) - 1u);
  return (referenced_outputs & required) == required;
}

} // namespace rund::compute::detail::device_vsm_product_detail

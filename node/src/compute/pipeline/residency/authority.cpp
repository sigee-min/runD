#include "authority.hpp"

#include "../../device/residency/pool.hpp"
#include "../state.hpp"

#include <algorithm>
#include <array>
#include <limits>

namespace rund::compute::detail {

bool has_private_residency_authority(const PipelineState &state) noexcept {
  if (state.device == nullptr || state.publication == nullptr ||
      state.residency == nullptr || state.transactional ||
      state.resources.empty() ||
      state.claims.size() != state.resources.size() ||
      !state.alternate_claims.empty()) {
    return false;
  }
  for (std::size_t index = 0u; index < state.resources.size(); ++index) {
    const PipelineResource &resource = state.resources[index];
    const BufferClaim &claim = state.claims[index];
    if (!resource.owned || resource.buffer == nullptr ||
        resource.buffer->device != state.device ||
        claim.buffer != resource.buffer.get() || claim.transactional_state ||
        claim.gated_publish || resource.buffer->poisoned) {
      return false;
    }
  }
  const bool graph = state.residency_port_count != 0u ||
                     state.residency_graph_stage != residency::NoGraphStage;
  if (graph) {
    const residency::Pool *const pool = state.residency_pool.get();
    if (pool == nullptr ||
        state.residency_graph_stage == residency::NoGraphStage ||
        state.residency_port_count == 0u ||
        state.residency_port_count > state.residency_ports.size() ||
        state.residency_control >= state.resources.size() ||
        state.residency_bank >= residency::Pool::BankCount ||
        pool->control[state.residency_bank] == nullptr) {
      return false;
    }
    std::array<bool, PipelineResourceCapacity> observed{};
    for (std::size_t index = 0u; index < state.residency_port_count; ++index) {
      const PipelineResidencyPort &port = state.residency_ports[index];
      if (port.pipeline_resource >= state.resources.size() ||
          port.pipeline_resource >= observed.size() ||
          observed[port.pipeline_resource] || port.page_bytes == 0u ||
          port.region.count == 0u ||
          port.remap_count > port.remaps.size()) {
        return false;
      }
      observed[port.pipeline_resource] = true;
      const PipelineResource &resource =
          state.resources[port.pipeline_resource];
      const BufferClaim &claim = state.claims[port.pipeline_resource];
      const bool write = residency::writes(port.access);
      if (claim.write != write || resource.bytes < port.page_bytes ||
          (write != (resource.output != PipelineResource::no_output))) {
        return false;
      }
    }
    const PipelineResource &control =
        state.resources[state.residency_control];
    const BufferClaim &control_claim = state.claims[state.residency_control];
    return !observed[state.residency_control] && control.owned &&
           control.output == PipelineResource::no_output &&
           !control_claim.write &&
           control.buffer == pool->control[state.residency_bank];
  }
  if (state.residency_graph_stage != residency::NoGraphStage ||
      state.residency_port_count != 0u ||
      state.residency_input >= state.resources.size() ||
      state.residency_output >= state.resources.size() ||
      state.residency_input == state.residency_output) {
    return false;
  }
  const PipelineResource &input = state.resources[state.residency_input];
  const PipelineResource &output = state.resources[state.residency_output];
  return input.output == PipelineResource::no_output &&
         output.output != PipelineResource::no_output &&
         state.claims[state.residency_input].write == false &&
         state.claims[state.residency_output].write;
}

} // namespace rund::compute::detail

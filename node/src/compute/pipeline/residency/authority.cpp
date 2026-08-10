#include "authority.hpp"

#include "../state.hpp"

#include <limits>

namespace rund::compute::detail {

bool has_private_residency_authority(const PipelineState &state) noexcept {
  if (state.device == nullptr || state.publication == nullptr ||
      state.residency == nullptr || state.transactional ||
      state.resources.empty() ||
      state.claims.size() != state.resources.size() ||
      !state.alternate_claims.empty() ||
      state.residency_input >= state.resources.size() ||
      state.residency_output >= state.resources.size() ||
      state.residency_input == state.residency_output) {
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
  const PipelineResource &input = state.resources[state.residency_input];
  const PipelineResource &output = state.resources[state.residency_output];
  return input.output == PipelineResource::no_output &&
         output.output != PipelineResource::no_output &&
         state.claims[state.residency_input].write == false &&
         state.claims[state.residency_output].write;
}

} // namespace rund::compute::detail

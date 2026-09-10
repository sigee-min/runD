#include "../claim.hpp"

namespace rund::compute::detail {

void close_pipeline_observation_epoch(PipelineState &state) noexcept {
  state.unobserved_outputs = 0u;
  state.observation_identity_valid = false;
  state.stats.output_hash = 0u;
  for (PipelineOutputState &output : state.outputs) {
    output.observed = false;
    output.hash = 0u;
  }
}

void synchronize_pipeline_observation_epoch(
    PipelineState &state,
    const PipelinePublicationState &publication) noexcept {
  if (state.observation_identity_valid &&
      (state.observation_generation != publication.generation ||
       state.observation_parity != publication.parity ||
       state.observation_payload_epoch != publication.payload_epoch)) {
    close_pipeline_observation_epoch(state);
  }
}

} // namespace rund::compute::detail

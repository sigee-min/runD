#include "internal.hpp"

namespace rund_node_test_pipeline_residency::service_free_direct_test {

bool product_success_evidence(const rund::compute::detail::PipelineState &state,
                              const std::uint64_t iterations) noexcept {
  const rund::compute::detail::ServiceFreeDirectProductEvidence &evidence =
      state.service_free_direct;
  return evidence.selected && !evidence.quarantined &&
         evidence.fixed_native_storage && evidence.fixed_common_storage &&
         evidence.iterations == iterations &&
         evidence.completed_iterations == iterations &&
         evidence.public_handoff_count == 1u &&
         evidence.native_submit_count == 1u &&
         evidence.epoch_native_submit_count == 0u &&
         evidence.payload_dispatch_count == 1u &&
         evidence.host_service_turn_count == 0u &&
         evidence.host_epoch_callback_count == 0u &&
         evidence.final_callback_count == 1u &&
         evidence.authority_publication_count == 1u &&
         evidence.registered_state_count != 0u;
}

} // namespace rund_node_test_pipeline_residency::service_free_direct_test

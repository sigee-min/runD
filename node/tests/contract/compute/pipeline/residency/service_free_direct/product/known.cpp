#include "internal.hpp"

#include "../actual.hpp"

namespace rund_node_test_pipeline_residency::service_free_direct_test {

bool product_known_rejection_retry(
    rund::compute::Pipeline &pipeline,
    const std::shared_ptr<rund::compute::detail::PipelineState> &state,
    const std::shared_ptr<
        const rund::node::accel::detail::ServiceFreeDirectProof>
        &proof) noexcept {
  using namespace rund::compute;
  if (state == nullptr || state->device == nullptr ||
      state->device->residency == nullptr || proof == nullptr) {
    return false;
  }
  ActualAuthorityRun occupied =
      begin_actual_authority_run(state->device->residency, proof);
  if (!occupied) {
    return false;
  }
  const std::uint64_t generation = pipeline.generation();
  const Status rejected = pipeline.run();
  const Stats rejected_stats = pipeline.stats();
  const bool rejected_exact =
      !rejected && rejected.reason() == Reason::PipelineInvalid &&
      pipeline.generation() == generation &&
      rejected_stats.command_submits == 0u && rejected_stats.dispatches == 0u &&
      state->service_free_direct.selected &&
      state->service_free_direct.public_handoff_count == 1u &&
      state->service_free_direct.native_submit_count == 0u &&
      state->service_free_direct.final_callback_count == 0u;
  const bool released =
      finish_actual_authority_run(occupied, Status::fail(Reason::PipelineBusy));
  const Status retried =
      released ? pipeline.run() : Status::fail(Reason::PipelineInvalid);
  return rejected_exact && retried &&
         pipeline.generation() == generation + 1u &&
         product_success_evidence(*state, proof->iterations);
}

} // namespace rund_node_test_pipeline_residency::service_free_direct_test

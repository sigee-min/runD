#include "local.hpp"

#include "src/compute/virtual/run/device_vsm/operations.hpp"

namespace rund_node_test_pipeline_residency::device_vsm_test {
namespace {

using rund::compute::detail::BufferState;
using rund::compute::detail::device_vsm_product_detail::
    DeviceVsmProductEvidence;
using rund::compute::detail::device_vsm_product_detail::DeviceVsmProductOwner;
using rund::compute::detail::device_vsm_product_detail::
    reset_device_vsm_run_evidence;

[[nodiscard]] std::shared_ptr<BufferState> ResidentToken() {
  return std::make_shared<BufferState>();
}

[[nodiscard]] bool ResidentCase(DeviceVsmProductOwner &owner) noexcept {
  owner.input_count = 2u;
  owner.resident_inputs[0u] = ResidentToken();
  owner.resident_inputs[1u].reset();
  owner.resident_output = ResidentToken();
  owner.cold_prepare_count = 7u;
  owner.warm_rearm_count = 11u;
  owner.output_hash_observation_count = 13u;
  owner.output_hash_reuse_count = 17u;
  owner.evidence = std::make_shared<DeviceVsmProductEvidence>();
  owner.evidence->native.native_submit_count = 99u;
  owner.evidence->final_received = true;
  owner.evidence->bounded_external_page_service = true;
  return reset_device_vsm_run_evidence(owner) &&
         owner.evidence->native.native_submit_count == 0u &&
         !owner.evidence->final_received &&
         owner.evidence->cold_prepare_count == 7u &&
         owner.evidence->warm_rearm_count == 11u &&
         owner.evidence->public_handoff_count == 1u &&
         owner.evidence->output_hash_observation_count == 13u &&
         owner.evidence->output_hash_reuse_count == 17u &&
         owner.evidence->public_resident_input_count == 1u &&
         owner.evidence->whole_run_staged_input_count == 1u &&
         owner.evidence->public_resident_output &&
         !owner.evidence->whole_run_staged_output &&
         !owner.evidence->bounded_external_page_service;
}

[[nodiscard]] bool StagedCase(DeviceVsmProductOwner &owner) noexcept {
  owner.resident_inputs[0u].reset();
  owner.resident_output.reset();
  return reset_device_vsm_run_evidence(owner) &&
         owner.evidence->public_resident_input_count == 0u &&
         owner.evidence->whole_run_staged_input_count == 2u &&
         !owner.evidence->public_resident_output &&
         owner.evidence->whole_run_staged_output &&
         !owner.evidence->bounded_external_page_service;
}

} // namespace

bool CheckProductEvidence() noexcept {
  DeviceVsmProductOwner owner{};
  if (reset_device_vsm_run_evidence(owner)) {
    return false;
  }
  return ResidentCase(owner) && StagedCase(owner);
}

} // namespace rund_node_test_pipeline_residency::device_vsm_test

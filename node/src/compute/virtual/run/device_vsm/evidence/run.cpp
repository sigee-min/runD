#include "../internal.hpp"

namespace rund::compute::detail::device_vsm_product_detail {

bool reset_device_vsm_run_evidence(DeviceVsmProductOwner &owner) noexcept {
  if (owner.evidence == nullptr) {
    return false;
  }
  *owner.evidence = DeviceVsmProductEvidence{
      .cold_prepare_count = owner.cold_prepare_count,
      .warm_rearm_count = owner.warm_rearm_count,
      .public_handoff_count = 1u,
      .output_hash_observation_count = owner.output_hash_observation_count,
      .output_hash_reuse_count = owner.output_hash_reuse_count,
  };
  classify_device_vsm_backing_evidence(owner, *owner.evidence);
  return true;
}

} // namespace rund::compute::detail::device_vsm_product_detail

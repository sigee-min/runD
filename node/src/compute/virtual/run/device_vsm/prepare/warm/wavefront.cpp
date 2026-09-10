#include "local.hpp"

namespace rund::compute::detail::device_vsm_product_detail::warm_detail {

bool same_wavefront(
    const node::accel::detail::DeviceVsmGraphWavefrontProof &left,
    const node::accel::detail::DeviceVsmGraphWavefrontProof &right) noexcept {
  if (left.stage_count != right.stage_count ||
      left.frame_capacity != right.frame_capacity ||
      left.batch_count != right.batch_count ||
      left.map_stage != right.map_stage ||
      left.collective_stage != right.collective_stage) {
    return false;
  }
  for (std::size_t stage = 0u;
       stage < node::accel::detail::DeviceVsmGraphStageCapacity; ++stage) {
    if (left.same_dispatch[stage] != right.same_dispatch[stage] ||
        left.same_release[stage] != right.same_release[stage] ||
        left.prior_dispatch[stage] != right.prior_dispatch[stage] ||
        left.prior_release[stage] != right.prior_release[stage]) {
      return false;
    }
  }
  return true;
}

} // namespace rund::compute::detail::device_vsm_product_detail::warm_detail

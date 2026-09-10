#include "projection.hpp"

#include "projection/internal.hpp"

namespace rund::node::accel::detail {

DeviceVsmProofProjection ProjectPreparedKernelPipelineDeviceVsm(
    const DeviceVsmProofRequest &request) noexcept {
  device_vsm_projection::Candidate candidate{};
  if (!device_vsm_projection::SelectProjectionCandidate(request, candidate)) {
    return {.check = candidate.check};
  }
  const rund::AccelCheck validation =
      device_vsm_projection::ValidateProjectionCandidate(request, candidate);
  if (!validation.ok) {
    return {.check = validation};
  }
  return device_vsm_projection::MaterializeProjection(request, candidate);
}

} // namespace rund::node::accel::detail

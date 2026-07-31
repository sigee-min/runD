#include <accel/device.hpp>
#include <accel/runtime.hpp>

#include "../backend/resource.hpp"

namespace rund::node::accel {

rund::RuntimeStats ReadRuntimeStats(const rund::AccelDevice &pick) {
  const detail::PickAdmission admission = detail::AdmitPick(pick);
  return admission.check.ok
             ? detail::ReadBackendStats(admission.token)
             : rund::RuntimeStats{.reason = "accel_buffer_backend_unavailable"};
}

void ResetRuntimeStats(const rund::AccelDevice &pick) {
  const detail::PickAdmission admission = detail::AdmitPick(pick);
  if (admission.check.ok) {
    detail::ResetBackendStats(admission.token);
  }
}

} // namespace rund::node::accel

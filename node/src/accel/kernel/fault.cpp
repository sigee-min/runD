#include "fault.hpp"

#include "../backend/token.hpp"

namespace rund::node::accel::detail {

bool InjectNativeDeviceLostOnce(const rund::AccelDevice &pick) noexcept {
  const PickAdmission admitted = AdmitPick(pick);
  const BackendOps *const ops = admitted.ops();
  const rund::AccelDevice *const canonical = admitted.raw();
  return admitted.check.ok && canonical != nullptr && ops != nullptr &&
         ops->inject_device_lost_once != nullptr &&
         ops->inject_device_lost_once(*canonical);
}

} // namespace rund::node::accel::detail

#include <accel/device.hpp>

#include "../backend/resource.hpp"

namespace rund::node::accel {

AccelMemoryStats ReadAccelMemoryStats(const rund::AccelDevice &pick) noexcept {
  const detail::PickAdmission admission = detail::AdmitPick(pick);
  return admission.check.ok ? detail::ReadBackendMemory(admission.token)
                            : AccelMemoryStats{};
}

} // namespace rund::node::accel

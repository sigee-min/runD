#include <accel/device.hpp>
#include <accel/runtime.hpp>

#include "../backend/resource.hpp"

#include <memory>

namespace rund::node::accel {

rund::RuntimeStats ReadRuntimeStats(const rund::AccelDevice &pick) {
  const std::shared_ptr<detail::PickToken> token = detail::AdmitPick(pick);
  return token != nullptr
             ? detail::ReadBackendStats(token)
             : rund::RuntimeStats{.reason = "accel_buffer_backend_unavailable"};
}

void ResetRuntimeStats(const rund::AccelDevice &pick) {
  const std::shared_ptr<detail::PickToken> token = detail::AdmitPick(pick);
  if (token != nullptr) {
    detail::ResetBackendStats(token);
  }
}

} // namespace rund::node::accel

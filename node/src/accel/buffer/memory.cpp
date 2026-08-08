#include <accel/device.hpp>

#include "../backend/resource.hpp"

#include <memory>

namespace rund::node::accel {

AccelMemoryStats ReadAccelMemoryStats(const rund::AccelDevice &pick) noexcept {
  const std::shared_ptr<detail::PickToken> token = detail::AdmitPick(pick);
  return token != nullptr ? detail::ReadBackendMemory(token)
                          : AccelMemoryStats{};
}

} // namespace rund::node::accel

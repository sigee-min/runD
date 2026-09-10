#include "local.hpp"

#include <cstdint>
#include <memory>

namespace rund::node::accel::detail {

std::uint64_t
MetalKernelTraffic(const std::shared_ptr<void> &prepared) noexcept {
#if defined(__APPLE__) && defined(RUND_NODE_HAVE_METAL_SDK)
  const auto *const resources =
      static_cast<const MetalKernelResources *>(prepared.get());
  return resources == nullptr ? 0u : resources->traffic;
#else
  (void)prepared;
  return 0u;
#endif
}

} // namespace rund::node::accel::detail

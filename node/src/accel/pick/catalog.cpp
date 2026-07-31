#include <node/accel/pick.hpp>

#include "../backend/catalog.hpp"
#include "../cpu/ops.hpp"
#include "../fake/ops.hpp"
#if defined(RUND_NODE_CATALOG_METAL)
#include "../metal/ops.hpp"
#endif
#if defined(RUND_NODE_CATALOG_VULKAN)
#include "../vulkan/ops.hpp"
#endif

namespace rund::node::accel {

rund::AccelDevice PickAccel(const rund::AccelPolicy &policy) {
  static const detail::BackendEntry catalog[]{
#if defined(RUND_NODE_CATALOG_METAL)
      detail::MetalEntry(),
#endif
#if defined(RUND_NODE_CATALOG_VULKAN)
      detail::VulkanEntry(),
#endif
      detail::FakeEntry(),
      detail::CpuEntry(),
  };
  return detail::PickFromCatalog(catalog, policy);
}

} // namespace rund::node::accel

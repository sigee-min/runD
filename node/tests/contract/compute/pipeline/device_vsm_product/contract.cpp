#include "local.hpp"

#if defined(RUND_NODE_TEST_BACKEND_CPU)

namespace rund_node_test_device_vsm_product {

int CheckDeviceVsmProduct(
    const rund::compute::Backend,
    const rund_node_test_persistent_product::NativeQueueCounter) noexcept {
  return 0;
}

} // namespace rund_node_test_device_vsm_product

#endif

#include "local.hpp"

namespace rund::node::accel::detail {

#if defined(RUND_NODE_HAVE_VULKAN_SDK)
void TryCreateVulkanTimestampQueryPool(VulkanAdapter& adapter) {
  adapter.timestamp_query_available = adapter.timestamp_valid_bits != 0u &&
                                      adapter.timestamp_period_ns > 0.0F;
  adapter.accel_timestamp_source = adapter.timestamp_query_available
                                       ? "vulkan_timestamp_query"
                                       : "unavailable";
}
#endif

}  // namespace rund::node::accel::detail

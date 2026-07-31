#include <accel/api.hpp>
#include <accel/check.hpp>
#include <accel/device.hpp>

#include "../local.hpp"

namespace rund::node::accel::detail {

rund::AccelDevice RejectVulkan(const char *const reason) {
  return rund::AccelDevice{
      .check = rund::AccelCheck{false, reason},
      .api = rund::AccelApi::Vulkan,
      .caps =
          rund::kernel::ComputeCaps{
              .api = rund::kernel::ComputeApi::Vulkan,
              .reason = reason,
          },
  };
}

} // namespace rund::node::accel::detail

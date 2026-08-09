#include <accel/check.hpp>

#include "../../stencil/metal.hpp"
#include "../range/local.hpp"

#include <string_view>

namespace rund::node::accel::detail {

rund::AccelCheck FinishMetalStencil(MetalAdapter &adapter,
                                    const std::shared_ptr<void> &resources) {
  const rund::AccelCheck check = FinishMetalRange(adapter, resources);
  if (!check.ok &&
      std::string_view{check.reason} == "compute_range_aggregate_invalid") {
    SetMetalLastError(adapter, "compute_stencil_invalid");
    return rund::AccelCheck{false, "compute_stencil_invalid"};
  }
  return check;
}

} // namespace rund::node::accel::detail

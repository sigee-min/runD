#include <accel/check.hpp>

#include "../../window/metal.hpp"
#include "../range/local.hpp"

#include <string_view>

namespace rund::node::accel::detail {

rund::AccelCheck FinishMetalWindow(MetalAdapter &adapter,
                                   const std::shared_ptr<void> &resources) {
  const rund::AccelCheck check = FinishMetalRange(adapter, resources);
  if (!check.ok &&
      std::string_view{check.reason} == "compute_range_aggregate_invalid") {
    SetMetalLastError(adapter, "compute_window_invalid");
    return rund::AccelCheck{false, "compute_window_invalid"};
  }
  return check;
}

} // namespace rund::node::accel::detail

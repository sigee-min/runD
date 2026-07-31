#include "local.hpp"
#include "../local/windows.hpp"

namespace node_accel_contract::runtime::window {

[[nodiscard]] std::vector<rund::kernel::ComputeDispatchWindow> DispatchWindows(
    const rund::kernel::ComputePlan& plan) {
  return runtime_case::DispatchWindows(plan);
}

}  // namespace node_accel_contract::runtime::window

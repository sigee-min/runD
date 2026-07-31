#include "solve/local.hpp"

#include "primitive/local.hpp"
#include <node/accel/pick.hpp>

#include <array>

namespace node_accel_contract {

[[nodiscard]] bool AvailableBackendsRunSolveNatively() {
  namespace fix = node_accel_contract::primitive;
  for (const rund::AccelApi api :
       {rund::AccelApi::Metal, rund::AccelApi::Vulkan}) {
    const rund::AccelDevice pick =
        rund::node::accel::PickAccel(fix::Policy(api));
    if (!pick.check.ok) {
      continue;
    }
    if (pick.api == rund::AccelApi::Cpu) {
      return false;
    }
    if (!BackendRunsSolve(pick) || !BackendRunsFactorReuseSolve(pick)) {
      return false;
    }
  }
  return true;
}

} // namespace node_accel_contract

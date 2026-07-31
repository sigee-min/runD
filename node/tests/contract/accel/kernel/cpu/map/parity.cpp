#include <accel/api.hpp>
#include <accel/device.hpp>

#include "../local.hpp"
#include "hash/run.hpp"
#include <node/accel/pick.hpp>

#include <array>

namespace node_accel_contract::cpu_context {

[[nodiscard]] bool
CpuContextMatchesAvailableRealBackend(const rund::AccelDevice &cpu_pick) {
  const MapRun cpu = ContextMapHash(cpu_pick);
  if (!cpu.ok) {
    return false;
  }

  const std::array<rund::AccelApi, 2u> real_apis{
      rund::AccelApi::Metal,
      rund::AccelApi::Vulkan,
  };
  bool saw_precise_unavailable = false;
  for (const rund::AccelApi api : real_apis) {
    const rund::AccelDevice pick = rund::node::accel::PickAccel(ApiPolicy(api));
    if (!pick.check.ok) {
      if (!PickUnavailableReasonIsPrecise(pick, api)) {
        return false;
      }
      saw_precise_unavailable = true;
      continue;
    }
    const MapRun real = ContextMapHash(pick);
    return real.ok && real.hash == cpu.hash;
  }
  return saw_precise_unavailable;
}

} // namespace node_accel_contract::cpu_context

#include <accel/device.hpp>

#include "run/body.hpp"

namespace node_accel_contract::cpu_context {

bool CpuContextRunsMap(const rund::AccelDevice &pick) {
  return CpuContextMapEvidenceCountersMatch(pick) &&
         CpuContextRejectsInvalidMapRun(pick);
}

} // namespace node_accel_contract::cpu_context

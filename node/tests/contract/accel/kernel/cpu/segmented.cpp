#include <accel/api.hpp>
#include <accel/device.hpp>

#include "segmented/run.hpp"

namespace node_accel_contract::cpu_context {

bool CpuContextRunsSegmentedScanThenMap(const rund::AccelDevice &pick) {
  return segmented::ContextRunsSegmentedScanThenMap(pick, rund::AccelApi::Cpu,
                                                    0u, 3u);
}

} // namespace node_accel_contract::cpu_context

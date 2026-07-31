#include "local.hpp"

#include <algorithm>
#include <cstddef>

namespace rund::kernel::schedule::planner {

u32 ClampKernelWidth(const u32 execution_width) {
  return std::max<u32>(1u, execution_width);
}

bool ValidPacketWorkUnits(const std::span<const u64> packet_work_units,
                          const PartitionRequest& request) {
  return request.packet_count != 0u &&
         request.execution_width != 0u &&
         packet_work_units.size() == static_cast<std::size_t>(request.packet_count);
}

} // namespace rund::kernel::schedule::planner

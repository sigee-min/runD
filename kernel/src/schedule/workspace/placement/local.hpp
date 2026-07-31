#pragma once

#include "../local.hpp"

namespace rund::kernel::workspace_placement {

u32 CountLocalityBucketCrossings(const ScheduleCompileRequest& request,
                                 std::span<const u32> ordered_packets);
const char* AffinityPlacementReason(const WorkerBackendCapabilities& capabilities);

} // namespace rund::kernel::workspace_placement

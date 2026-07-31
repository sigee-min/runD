#pragma once

#include "local.hpp"

namespace rund::kernel::program_detail {

KernelProgramTilePlan BuildKernelProgramTilePlan(
    ScheduleView schedule,
    const WorkerBackendCapabilities& capabilities,
    const KernelProgramPhysicalTilePolicy& physical_tile_policy);

} // namespace rund::kernel::program_detail

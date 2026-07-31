#pragma once

#include "../state.hpp"

namespace rund::kernel::program_detail {

TimePoint Now();
void RecordProgramCompileCost(Workspace& workspace, TimePoint start);
void RecordCapacityCheckCost(Workspace& workspace, TimePoint start);
void RecordScheduleCompileCost(Workspace& workspace, TimePoint start);
void RecordFoldGraphCompileCost(Workspace& workspace, TimePoint start);
void RecordPlacementCost(Workspace& workspace, TimePoint start);

} // namespace rund::kernel::program_detail

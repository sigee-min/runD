#pragma once

#include "local.hpp"

namespace rund::kernel::workspace_detail {

void BuildWeightedPacketOrder(Workspace& workspace,
                              const ScheduleCompileRequest& request,
                              std::span<const u64> resolved_work_units);

} // namespace rund::kernel::workspace_detail

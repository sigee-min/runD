#pragma once

#include "../local.hpp"

namespace rund::kernel::workspace_detail {

void AssignEightPartitions(Workspace &workspace,
                           std::span<const u64> resolved_work_units);

} // namespace rund::kernel::workspace_detail

#pragma once

#include "../state.hpp"

#include <kernel/program/request.hpp>

namespace rund::kernel::program_detail {

const char* ValidateDispatchRequirements(const KernelProgramCompileRequest& request,
                                         const ScheduleCompileRequest& schedule_request,
                                         const Workspace& workspace,
                                         const WorkerBackendCapabilities& capabilities);

} // namespace rund::kernel::program_detail

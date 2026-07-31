#pragma once

#include "../state.hpp"

#include <kernel/reduction/fold/primitive.hpp>

namespace rund::kernel::program_detail {

void AttachStrictFloatTelemetry(Telemetry &telemetry, FoldOperation operation,
                                StrictFloatReductionPolicy policy,
                                const WorkerBackendCapabilities &capabilities);

} // namespace rund::kernel::program_detail

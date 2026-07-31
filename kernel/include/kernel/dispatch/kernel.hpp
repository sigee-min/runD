#pragma once

#include <kernel/dispatch/telemetry.hpp>
#include <kernel/dispatch/worker/backend.hpp>

namespace rund::kernel {

WorkerBackendCapabilities InspectWorkerBackend(const WorkerBackend& backend, u32 requested_width);

} // namespace rund::kernel

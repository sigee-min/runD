#pragma once

#include "../local.hpp"

namespace rund::kernel::dispatch::detail::serial {

void ApplySingleWorkerStats(std::span<u32> worker_stats_sink,
                            bool require_no_allocation,
                            u32 executed_partitions,
                            Telemetry& telemetry);

Result BuildSerialResult(bool ok, Telemetry&& telemetry);

} // namespace rund::kernel::dispatch::detail::serial

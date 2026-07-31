#pragma once

#include <kernel/dispatch/orchestrator/request.hpp>
#include <kernel/dispatch/orchestrator/result.hpp>

namespace rund::kernel {

void ResetFailureSignal(FailureSignal &signal);
void MarkFailure(FailureSignal &signal, const char *reason);
bool HasFailure(const FailureSignal &signal);
RunResult RunPreparedProgram(const RunPreparedProgramRequest &request);

} // namespace rund::kernel

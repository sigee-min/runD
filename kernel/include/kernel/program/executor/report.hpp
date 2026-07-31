#pragma once

#include <kernel/program/executor/model.hpp>
#include <kernel/program/report.hpp>

namespace rund::kernel {

[[nodiscard]] inline KernelExecutionReport execution_report(
    const Executor& exec) noexcept {
  if (!exec.valid) {
    return invalid_execution_report(exec.reason);
  }
  if (exec.workspace == nullptr) {
    return invalid_execution_report(exec.reason);
  }
  return execution_report(*exec.workspace);
}

} // namespace rund::kernel

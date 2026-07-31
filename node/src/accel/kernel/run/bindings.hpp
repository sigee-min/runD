#pragma once

#include <accel/kernel/run.hpp>

#include "../backend/run.hpp"

namespace rund::node::accel::detail {

struct RunBindBuild {
  RunBinds binds{};
  bool ok = false;
  const char *reason = "accel_kernel_run_invalid";
};

struct ResetBindBuild {
  BoundResets binds{};
  bool ok = false;
  const char *reason = "accel_kernel_reset_invalid";
};

[[nodiscard]] RunBindBuild BuildRunBinds(const KernelExecution &execution,
                                         const rund::AccelRun &run);

[[nodiscard]] ResetBindBuild BuildResetBinds(const KernelExecution &execution,
                                             const rund::AccelRun &run,
                                             const RunBinds &bindings);

} // namespace rund::node::accel::detail

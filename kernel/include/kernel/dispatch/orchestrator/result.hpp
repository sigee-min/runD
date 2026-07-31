#pragma once

#include <kernel/dispatch/telemetry.hpp>

namespace rund::kernel {

struct RunResult {
  bool ok = false;
  const char *reason = "not_run";
  bool domain_failed = false;
  Result kernel{};
};

} // namespace rund::kernel

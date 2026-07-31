#pragma once

#include <accel/api.hpp>
#include <accel/runtime.hpp>
#include <kernel/program/compute/model.hpp>

#include <cstdint>

namespace rund {

struct AccelContextEvidence {
  AccelApi api = AccelApi::Auto;
  rund::kernel::ComputeCaps caps{};
  RuntimeStats runtime_stats{};
  bool ok = false;
  const char* reason = "accel_context_invalid";
};

}  // namespace rund

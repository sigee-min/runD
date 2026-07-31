#pragma once

#include <accel/check.hpp>
#include <accel/runtime.hpp>

#include "status.hpp"

namespace rund::node::accel::detail {

struct KernelResult final {
  rund::AccelCheck check{};
  rund::RuntimeStats stats{};
  PreparedPipelineBackendEvidence pipeline{};
};

using KernelCompletion = void (*)(void *, KernelResult) noexcept;

} // namespace rund::node::accel::detail

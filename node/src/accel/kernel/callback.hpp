#pragma once

#include <accel/check.hpp>
#include <accel/runtime.hpp>

#include "status.hpp"

#include <cstdint>

namespace rund::node::accel::detail {

enum class KernelTiming : std::uint8_t {
  None,
  Submission,
  Dispatch,
};

struct KernelResult final {
  rund::AccelCheck check{};
  rund::RuntimeStats stats{};
  PreparedPipelineBackendEvidence pipeline{};
};

using KernelCompletion = void (*)(void *, KernelResult) noexcept;

} // namespace rund::node::accel::detail

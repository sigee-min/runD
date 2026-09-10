#pragma once

#include <accel/check.hpp>
#include <accel/runtime.hpp>

#include "status.hpp"
#include "terminal.hpp"

#include <cstdint>

namespace rund::node::accel::detail {

enum class KernelTiming : std::uint8_t {
  None,
  Submission,
  Dispatch,
};

enum class PipelineSubmitMode : std::uint8_t {
  Standard,
  Residency,
};

struct KernelResult final {
  rund::AccelCheck check{};
  rund::RuntimeStats stats{};
  PreparedPipelineBackendEvidence pipeline{};
  NativeTerminal terminal{NativeTerminal::Known};
};

using KernelCompletion = void (*)(void *, KernelResult) noexcept;

} // namespace rund::node::accel::detail

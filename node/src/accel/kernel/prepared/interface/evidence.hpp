#pragma once

#include "../../profile.hpp"
#include "../../status.hpp"
#include "../../terminal.hpp"
#include <accel/check.hpp>
#include <accel/kernel/evidence.hpp>
#include <cstdint>

namespace rund::node::accel::detail {

struct PreparedBatchEvidence final {
  rund::AccelEvidence shared{};
  rund::AccelCheck check{};
};

struct PreparedPipelineEvidence final {
  rund::AccelEvidence shared{};
  rund::AccelCheck check{};
  NativeTerminal terminal{NativeTerminal::Known};
  PreparedPipelineControl control{};
  PreparedPipelineProfileEvidence profile{};
  std::uint64_t status_entry_count{};
  std::uint64_t control_byte_count{};
  std::uint64_t control_command_count{};
  std::uint64_t control_ns{};
  std::uint32_t active_step_count{};
  bool submitted{};
  bool control_observed{};
  bool control_valid{};
};


} // namespace rund::node::accel::detail

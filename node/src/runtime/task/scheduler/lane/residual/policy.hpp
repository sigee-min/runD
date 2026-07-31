#pragma once

#include <cstdint>
#include <string_view>

namespace rund::node {

struct LaneResidualPolicyInput {
  std::uint32_t participating_lanes = 0u;
  std::uint32_t logical_tasks = 0u;
  bool all_success_terminal = false;
  bool canonical_home_lanes_preserved = false;
  bool has_side_exit = false;
};

struct LaneResidualPolicyDecision {
  bool accepted = false;
  std::string_view reason{};
};

[[nodiscard]] LaneResidualPolicyDecision DecideLaneResidualPolicy(
    LaneResidualPolicyInput input) noexcept;

}  // namespace rund::node

#include "policy.hpp"

namespace rund::node {

LaneResidualPolicyDecision DecideLaneResidualPolicy(
    const LaneResidualPolicyInput input) noexcept {
  if (input.has_side_exit) {
    return LaneResidualPolicyDecision{
        .accepted = false,
        .reason = "lane_residual_side_exit",
    };
  }
  if (!input.canonical_home_lanes_preserved) {
    return LaneResidualPolicyDecision{
        .accepted = false,
        .reason = "lane_residual_home_lane_unproved",
    };
  }
  if (!input.all_success_terminal) {
    return LaneResidualPolicyDecision{
        .accepted = false,
        .reason = "lane_residual_not_all_success",
    };
  }
  if (input.participating_lanes <= 1u) {
    return LaneResidualPolicyDecision{
        .accepted = false,
        .reason = "lane_residual_single_lane",
    };
  }
  return LaneResidualPolicyDecision{
      .accepted = true,
      .reason = "ok",
  };
}

}  // namespace rund::node

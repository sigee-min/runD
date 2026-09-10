#pragma once

#include "model/frame.hpp"

namespace rund::compute::detail::residency::frame_detail {

[[nodiscard]] inline registry_model::Frame
empty(const registry_model::Frame &frame) noexcept {
  return registry_model::Frame{.tier = frame.tier,
                               .role = frame.role,
                               .extent = frame.extent,
                               .view = frame.view,
                               .assigned = frame.assigned,
                               .transaction_generation = 0u,
                               .view_commit_stamp = 0u};
}

[[nodiscard]] inline registry_model::Frame
stamp(const registry_model::Frame &frame, const std::uint64_t value) noexcept {
  registry_model::Frame result = frame;
  result.view_commit_stamp = value;
  return result;
}

} // namespace rund::compute::detail::residency::frame_detail

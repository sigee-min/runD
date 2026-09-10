#pragma once

#include "internal.hpp"

namespace rund::compute::detail::residency::physical::promote {

template <typename State>
[[nodiscard]] bool
validate_projection(const execution::Plan &plan, State &state,
                    const execution::SlidingProjection &projection,
                    const std::span<PageUse> uses,
                    const std::size_t frame_count,
                    Projection &result) noexcept {
  execution::SlidingProjection expected{};
  execution::Node input{};
  execution::Node dispatch{};
  execution::Node output{};
  execution::WindowFootprintProjection footprint{};
  std::array<execution::SlidingCoordinate,
             execution::SlidingPredecessorCapacity>
      predecessors{};
  std::size_t predecessor_count = 0u;
  const bool window = plan.has_window_footprint();
  if (!state.exact(projection, uses, expected) ||
      !state.invocation.predecessors(projection, predecessors,
                                     predecessor_count) ||
      !plan.project(execution::NodeId{.epoch = projection.coordinate.ordinal,
                                      .phase = execution::Phase::Input},
                    input) ||
      !plan.project(execution::NodeId{.epoch = projection.coordinate.ordinal,
                                      .phase = execution::Phase::Dispatch},
                    dispatch) ||
      !plan.project(execution::NodeId{.epoch = projection.coordinate.ordinal,
                                      .phase = execution::Phase::Output},
                    output) ||
      (window &&
       !plan.window_footprint(projection.coordinate.ordinal, footprint)) ||
      (window ? footprint.source_count : input.input_count) !=
          projection.fetch_count ||
      dispatch.input_count !=
          (window ? footprint.target_count : input.input_count) ||
      dispatch.output_count != projection.persist_count ||
      output.output_count != dispatch.output_count ||
      (window ? footprint.source_count : input.input_count) >
          execution::WindowFootprintSourceCapacity ||
      dispatch.input_count > execution::UseCapacity ||
      output.output_count > execution::UseCapacity) {
    return false;
  }
  for (std::size_t index = 0u; index < predecessor_count; ++index) {
    if (!state.terminal_for(predecessors[index])) {
      return false;
    }
  }
  const std::size_t bank = state.host_bank(projection.coordinate);
  if (bank >= execution::BankCapacity) {
    return false;
  }
  const FrameRegion device_input = plan.device_input_regions()[bank];
  const FrameRegion device_output = plan.device_output_regions()[bank];
  const FrameRegion host_output = plan.host_output_regions()[bank];
  const auto valid_region = [&](const FrameRegion region, const FrameTier tier,
                                const FrameRole role) noexcept {
    return region.tier == tier && region.role == role &&
           region.first <= frame_count &&
           region.count <= frame_count - region.first;
  };
  if (!valid_region(device_input, FrameTier::Device, FrameRole::Input) ||
      !valid_region(device_output, FrameTier::Device, FrameRole::Output) ||
      !valid_region(host_output, FrameTier::Host, FrameRole::Output) ||
      dispatch.input_count > device_input.count ||
      output.output_count > device_output.count ||
      state.output_stride > host_output.count) {
    return false;
  }
  result = Projection{.input = input,
                      .output = output,
                      .footprint = footprint,
                      .device_input = device_input,
                      .device_output = device_output,
                      .host_output = host_output,
                      .assembles_window = window};
  return true;
}

} // namespace rund::compute::detail::residency::physical::promote

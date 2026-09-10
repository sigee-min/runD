#pragma once

#include "internal.hpp"

namespace rund::compute::detail::residency::physical::promote {

inline void mint_ticket(
    const Identity plan, const std::uint64_t token,
    const std::uint64_t generation, const std::uint64_t owner,
    const execution::SlidingProjection &projection, const Selection &selected,
    const Projection &validated, const Commit committed,
    execution::SlidingTicket &credential,
    std::array<std::uint32_t, execution::WindowFootprintSourceCapacity>
        &host_frames,
    std::array<std::uint32_t, execution::UseCapacity> &device_input_frames,
    std::array<std::uint32_t, execution::UseCapacity> &device_output_frames,
    std::array<std::uint32_t, execution::UseCapacity> &host_output_frames,
    std::array<execution::FetchSource, execution::UseCapacity> &targets,
    std::array<execution::WindowFootprintSlice,
               execution::WindowFootprintSliceCapacity> &slices,
    std::uint64_t &nonce, std::uint32_t &transfer_mask,
    std::uint32_t &source_count, std::uint32_t &input_count,
    std::uint32_t &output_count, std::uint32_t &slice_count) noexcept {
  credential = execution::SlidingTicket{
      .plan = plan,
      .coordinate = projection.coordinate,
      .token = token,
      .generation = generation,
      .owner = owner,
      .turn = committed.turn,
      .expected_bytes = selected.transfer_bytes,
      .slot = static_cast<std::uint32_t>(projection.coordinate.ordinal %
                                         execution::SlidingNativeCapacity),
      .kind = execution::SlidingTicketKind::Promote,
  };
  host_frames = selected.host_frames;
  device_input_frames = selected.device_input_frames;
  device_output_frames = selected.device_output_frames;
  host_output_frames = selected.host_output_frames;
  if (validated.assembles_window) {
    targets = validated.footprint.targets;
    slices = validated.footprint.slices;
  }
  nonce = committed.turn;
  transfer_mask = selected.transfer_mask;
  source_count = committed.source_count;
  input_count = committed.input_count;
  output_count = committed.output_count;
  slice_count =
      validated.assembles_window
          ? static_cast<std::uint32_t>(validated.footprint.slice_count)
          : 0u;
}

} // namespace rund::compute::detail::residency::physical::promote

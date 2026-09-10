#pragma once

#include "internal.hpp"

namespace rund::compute::detail::residency::physical::promote {

template <typename State>
[[nodiscard]] Commit
commit_state(State &state, const std::span<Authority::Frame> frames,
             const execution::SlidingProjection &projection,
             const Projection &validated, const Selection &selected) noexcept {
  const auto map = [](Authority::Frame &frame, const CacheKey key,
                      const std::uint64_t next_use,
                      const std::uint64_t retain_until) noexcept {
    const Authority::Frame prior = frame;
    frame = Authority::Frame{.key = key,
                             .next_use = next_use,
                             .retain_until = retain_until,
                             .state = FrameState::Mapping,
                             .tier = prior.tier,
                             .role = prior.role,
                             .extent = prior.extent,
                             .view = prior.view,
                             .assigned = prior.assigned};
  };
  auto input_ring = state.input_ring(projection.coordinate);
  auto output_ring = state.output_ring(projection.coordinate);
  const std::size_t source_count = validated.assembles_window
                                       ? validated.footprint.source_count
                                       : validated.input.input_count;
  const std::size_t input_count = validated.assembles_window
                                      ? validated.footprint.target_count
                                      : validated.input.input_count;
  for (std::size_t local = 0u; local < source_count; ++local) {
    auto &input = input_ring[selected.input_slots[local]];
    input.state = execution::InputState::Promoting;
  }
  for (std::size_t local = 0u; local < input_count; ++local) {
    if ((selected.transfer_mask & (std::uint32_t{1u} << local)) != 0u) {
      const CacheUse &demand = validated.input.input[local];
      map(frames[selected.device_input_frames[local]],
          validated.assembles_window ? validated.footprint.targets[local].key
                                     : demand.key,
          demand.next_use, demand.retain_until);
    }
  }
  for (std::size_t local = 0u; local < validated.output.output_count; ++local) {
    const std::size_t use = source_count + local;
    auto &cell = output_ring[selected.output_slots[local]];
    cell.coordinate = projection.coordinate;
    cell.turn = cell.turn == std::numeric_limits<std::uint64_t>::max()
                    ? 1u
                    : cell.turn + 1u;
    cell.expected_bytes = validated.output.output[local].dirty.bytes;
    cell.physical_frame = selected.host_output_frames[local];
    cell.use = static_cast<std::uint32_t>(use);
    cell.state = execution::OutputState::Reserved;
    cell.completion = Status::success();
    cell.completion_terminal = execution::TerminalKind::Known;
    cell.completion_bytes = 0u;
    cell.completion_may_write = false;
    cell.physical_handoff = false;
    cell.invalidate_after_terminal = false;
  }
  auto &native = state.native_cells[static_cast<std::size_t>(
      projection.coordinate.ordinal % execution::SlidingNativeCapacity)];
  native.coordinate = projection.coordinate;
  native.host_frames = selected.host_frames;
  native.device_input_frames = selected.device_input_frames;
  native.device_output_frames = selected.device_output_frames;
  native.turn = native.turn == std::numeric_limits<std::uint64_t>::max()
                    ? 1u
                    : native.turn + 1u;
  native.promote_bytes = selected.transfer_bytes;
  native.transfer_mask = selected.transfer_mask;
  native.source_count = static_cast<std::uint32_t>(source_count);
  native.input_count = static_cast<std::uint32_t>(input_count);
  native.output_count =
      static_cast<std::uint32_t>(validated.output.output_count);
  native.resolved = 0u;
  native.state = execution::NativeState::Promoting;
  native.completion = Status::success();
  native.completion_terminal = execution::TerminalKind::Known;
  native.completion_bytes = 0u;
  native.completion_may_write = false;
  native.physical_handoff = true;
  native.suppress = false;
  native.invalidate_after_terminal = false;
  ++state.promote_calls;
  return Commit{.turn = native.turn,
                .source_count = native.source_count,
                .input_count = native.input_count,
                .output_count = native.output_count};
}

} // namespace rund::compute::detail::residency::physical::promote

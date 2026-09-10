#pragma once

#include "internal.hpp"

namespace rund::compute::detail::residency::physical::promote {

template <typename State>
[[nodiscard]] bool select_inputs(const execution::Plan &plan, State &state,
                                 const std::span<const Authority::Frame> frames,
                                 const execution::SlidingProjection &projection,
                                 const Projection &validated,
                                 Selection &result) noexcept {
  const auto ring = state.input_ring(projection.coordinate);
  const std::size_t source_count = validated.assembles_window
                                       ? validated.footprint.source_count
                                       : validated.input.input_count;
  for (std::size_t local = 0u; local < source_count; ++local) {
    const auto found =
        std::find_if(ring.begin(), ring.end(), [&](const auto &cell) {
          return cell.state == execution::InputState::Ready &&
                 !cell.physical_handoff &&
                 cell.coordinate == projection.coordinate && cell.use == local;
        });
    execution::FetchSource source{};
    const CacheUse &demand = validated.assembles_window
                                 ? validated.footprint.sources[local]
                                 : validated.input.input[local];
    if (found == ring.end() ||
        !(validated.assembles_window
              ? plan.canonical_input_source(demand.key.page, source)
              : plan.input_source(demand.key.page, source)) ||
        !source.materializes_frame() ||
        found->physical_frame >= frames.size()) {
      return false;
    }
    const Authority::Frame &host = frames[found->physical_frame];
    if (!host.assigned || host.tier != FrameTier::Host ||
        host.role != FrameRole::Input || host.state != FrameState::Resident ||
        !host.dirty.empty() || host.key != demand.key) {
      return false;
    }
    result.input_slots[local] =
        static_cast<std::uint32_t>(found - ring.begin());
    result.host_frames[local] = found->physical_frame;
  }
  const std::size_t input_count = validated.assembles_window
                                      ? validated.footprint.target_count
                                      : validated.input.input_count;
  for (std::size_t local = 0u; local < input_count; ++local) {
    execution::FetchSource source{};
    const CacheUse &demand = validated.input.input[local];
    if (validated.assembles_window) {
      source = validated.footprint.targets[local];
    } else if (!plan.input_source(demand.key.page, source)) {
      return false;
    }
    const std::uint32_t target =
        validated.device_input.first + static_cast<std::uint32_t>(local);
    const Authority::Frame &device = frames[target];
    const CacheKey target_key =
        validated.assembles_window ? source.key : demand.key;
    if (!source.materializes_frame() || !device.assigned ||
        device.tier != FrameTier::Device || device.role != FrameRole::Input ||
        !device.dirty.empty() ||
        (device.state != FrameState::Empty &&
         device.state != FrameState::Resident) ||
        (device.state == FrameState::Resident && device.key != target_key &&
         device.retain_until != NeverUse &&
         device.retain_until >= projection.coordinate.ordinal)) {
      return false;
    }
    result.device_input_frames[local] = target;
    if (device.state != FrameState::Resident || device.key != target_key) {
      result.transfer_mask |= std::uint32_t{1u} << local;
      if (source.frame_bytes >
          std::numeric_limits<std::uint64_t>::max() - result.transfer_bytes) {
        return false;
      }
      result.transfer_bytes += source.frame_bytes;
    }
  }
  return true;
}

template <typename State>
[[nodiscard]] bool
select_outputs(State &state, const std::span<const Authority::Frame> frames,
               const execution::SlidingProjection &projection,
               const std::span<PageUse> uses, const Projection &validated,
               Selection &result) noexcept {
  std::array<bool, execution::SlidingHostCapacity> selected_outputs{};
  const auto ring = state.output_ring(projection.coordinate);
  for (std::size_t local = 0u; local < validated.output.output_count; ++local) {
    const std::size_t use =
        (validated.assembles_window ? validated.footprint.source_count
                                    : validated.input.input_count) +
        local;
    const auto found =
        std::find_if(ring.begin(), ring.end(), [&](const auto &cell) {
          const std::size_t slot =
              static_cast<std::size_t>(&cell - ring.data());
          return cell.state == execution::OutputState::Free &&
                 !selected_outputs[slot];
        });
    if (found == ring.end() || use >= uses.size()) {
      return false;
    }
    const std::uint32_t device_frame =
        validated.device_output.first + static_cast<std::uint32_t>(local);
    const std::uint32_t output_slot =
        static_cast<std::uint32_t>(found - ring.begin());
    const std::uint32_t host_frame = validated.host_output.first + output_slot;
    const Authority::Frame &device = frames[device_frame];
    const Authority::Frame &host = frames[host_frame];
    if (!device.assigned || device.tier != FrameTier::Device ||
        device.role != FrameRole::Output || !device.dirty.empty() ||
        (device.state != FrameState::Empty &&
         device.state != FrameState::Resident) ||
        (device.state == FrameState::Resident &&
         device.retain_until != NeverUse &&
         device.retain_until >= projection.coordinate.ordinal) ||
        !host.assigned || host.tier != FrameTier::Host ||
        host.role != FrameRole::Output || !host.dirty.empty() ||
        (host.state != FrameState::Empty &&
         host.state != FrameState::Resident) ||
        (host.state == FrameState::Resident && host.retain_until != NeverUse &&
         host.retain_until >= projection.coordinate.ordinal)) {
      return false;
    }
    result.output_slots[local] = output_slot;
    selected_outputs[output_slot] = true;
    result.device_output_frames[local] = device_frame;
    result.host_output_frames[local] = host_frame;
  }
  return true;
}

template <typename State>
[[nodiscard]] bool
native_available(const State &state,
                 const execution::SlidingCoordinate coordinate) noexcept {
  const auto &native = state.native_cells[static_cast<std::size_t>(
      coordinate.ordinal % execution::SlidingNativeCapacity)];
  return native.state == execution::NativeState::Free &&
         state.promote_calls != std::numeric_limits<std::uint64_t>::max();
}

template <typename State>
[[nodiscard]] bool
select_physical(const execution::Plan &plan, State &state,
                const std::span<const Authority::Frame> frames,
                const execution::SlidingProjection &projection,
                const std::span<PageUse> uses, const Projection &validated,
                Selection &result) noexcept {
  return select_inputs(plan, state, frames, projection, validated, result) &&
         select_outputs(state, frames, projection, uses, validated, result) &&
         native_available(state, projection.coordinate);
}

} // namespace rund::compute::detail::residency::physical::promote

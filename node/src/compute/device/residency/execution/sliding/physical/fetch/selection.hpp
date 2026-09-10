#pragma once

#include "internal.hpp"

namespace rund::compute::detail::residency::physical::fetch {

template <typename State, typename Ring>
[[nodiscard]] std::uint64_t
consumer_horizon(const State &state,
                 const std::span<const Authority::Frame> frames,
                 const FrameRegion region, const Ring ring,
                 const std::size_t slot) noexcept {
  const auto &cell = ring[slot];
  return cell.state == execution::InputState::Ready &&
                 cell.coordinate.ordinal >= state.admitted
             ? cell.coordinate.ordinal
             : frames[static_cast<std::size_t>(region.first) + slot].next_use;
}

template <typename State, typename Ring>
[[nodiscard]] bool
eligible_victim(const State &state,
                const std::span<const Authority::Frame> frames,
                const execution::SlidingProjection &projection,
                const FrameRegion region, const Ring ring,
                const std::uint64_t frontier, const std::size_t slot) noexcept {
  const auto &cell = ring[slot];
  const Authority::Frame &frame =
      frames[static_cast<std::size_t>(region.first) + slot];
  const bool represented = cell.state == execution::InputState::Ready &&
                           cell.coordinate.ordinal != frontier &&
                           cell.physical_frame == region.first + slot &&
                           cell.pin.last_epoch < projection.coordinate.ordinal;
  const bool inherited = cell.state == execution::InputState::Free &&
                         (frame.retain_until == NeverUse ||
                          frame.retain_until < projection.coordinate.ordinal);
  return (represented || inherited) && frame.assigned &&
         frame.tier == FrameTier::Host && frame.role == FrameRole::Input &&
         frame.state == FrameState::Resident && frame.dirty.empty() &&
         (frame.retain_until == NeverUse ||
          frame.retain_until < projection.coordinate.ordinal) &&
         (projection.coordinate.ordinal == frontier ||
          consumer_horizon(state, frames, region, ring, slot) >
              projection.coordinate.ordinal);
}

template <typename Ring>
[[nodiscard]] std::size_t
find_hit(const std::span<const Authority::Frame> frames,
         const FrameRegion region, const Ring ring,
         const CacheKey key) noexcept {
  for (std::size_t slot = 0u; slot < ring.size(); ++slot) {
    const Authority::Frame &frame =
        frames[static_cast<std::size_t>(region.first) + slot];
    const auto &cell = ring[slot];
    if (frame.assigned && frame.state == FrameState::Resident &&
        frame.dirty.empty() && frame.key == key &&
        (cell.state == execution::InputState::Free ||
         (cell.state == execution::InputState::Ready &&
          cell.physical_frame == region.first + slot))) {
      return slot;
    }
  }
  return ring.size();
}

template <typename Ring>
[[nodiscard]] bool
future_hit_exists(const std::span<const Authority::Frame> frames,
                  const FrameRegion region, const Ring ring,
                  const CacheKey key) noexcept {
  for (std::size_t slot = 0u; slot < ring.size(); ++slot) {
    const Authority::Frame &frame =
        frames[static_cast<std::size_t>(region.first) + slot];
    if (frame.assigned && frame.state == FrameState::Resident &&
        frame.dirty.empty() && frame.key == key) {
      return true;
    }
  }
  return false;
}

template <typename State, typename Ring>
[[nodiscard]] bool future_capacity_available(
    const State &state, const std::span<const Authority::Frame> frames,
    const execution::SlidingProjection &projection, const FrameRegion region,
    const Ring ring, const std::uint64_t frontier) noexcept {
  const std::size_t represented = static_cast<std::size_t>(
      std::count_if(ring.begin(), ring.end(), [&](const auto &cell) {
        return cell.state != execution::InputState::Free &&
               cell.coordinate.ordinal == frontier;
      }));
  std::size_t available = 0u;
  for (std::size_t slot = 0u; slot < ring.size(); ++slot) {
    const Authority::Frame &frame =
        frames[static_cast<std::size_t>(region.first) + slot];
    if ((ring[slot].state == execution::InputState::Free && frame.assigned &&
         frame.state == FrameState::Empty) ||
        eligible_victim(state, frames, projection, region, ring, frontier,
                        slot)) {
      ++available;
    }
  }
  const std::size_t reserved = state.coordinate_input_count > represented
                                   ? state.coordinate_input_count - represented
                                   : 0u;
  return available > reserved;
}

template <typename Ring>
[[nodiscard]] std::size_t
find_empty(const std::span<const Authority::Frame> frames,
           const FrameRegion region, const Ring ring) noexcept {
  for (std::size_t slot = 0u; slot < ring.size(); ++slot) {
    const Authority::Frame &frame =
        frames[static_cast<std::size_t>(region.first) + slot];
    if (ring[slot].state == execution::InputState::Free && frame.assigned &&
        frame.state == FrameState::Empty) {
      return slot;
    }
  }
  return ring.size();
}

template <typename State, typename Ring>
[[nodiscard]] std::size_t find_farthest_victim(
    const State &state, const std::span<const Authority::Frame> frames,
    const execution::SlidingProjection &projection, const FrameRegion region,
    const Ring ring, const std::uint64_t frontier) noexcept {
  std::size_t selected = ring.size();
  for (std::size_t slot = 0u; slot < ring.size(); ++slot) {
    if (eligible_victim(state, frames, projection, region, ring, frontier,
                        slot) &&
        (selected == ring.size() ||
         consumer_horizon(state, frames, region, ring, slot) >
             consumer_horizon(state, frames, region, ring, selected))) {
      selected = slot;
    }
  }
  return selected;
}

[[nodiscard]] inline std::uint32_t find_reuse_frame(
    const std::span<const Authority::Frame> frames,
    const std::array<FrameRegion, execution::BankCapacity> &regions,
    const std::size_t active_banks, const CacheKey key,
    const std::uint32_t target) noexcept {
  for (std::size_t bank = 0u; bank < active_banks; ++bank) {
    const FrameRegion region = regions[bank];
    if (region.tier != FrameTier::Host || region.role != FrameRole::Input ||
        region.first > frames.size() ||
        region.count > frames.size() - region.first) {
      continue;
    }
    for (std::uint32_t local = 0u; local < region.count; ++local) {
      const std::uint32_t physical = region.first + local;
      const Authority::Frame &frame = frames[physical];
      if (physical != target && frame.assigned &&
          frame.state == FrameState::Resident && frame.dirty.empty() &&
          frame.key == key) {
        return physical;
      }
    }
  }
  return std::numeric_limits<std::uint32_t>::max();
}

template <typename State>
[[nodiscard]] bool
select_physical(State &state, const std::span<const Authority::Frame> frames,
                const execution::SlidingProjection &projection,
                const Projection &validated, const std::size_t active_banks,
                Selection &result) noexcept {
  const auto ring = state.input_ring(projection.coordinate);
  const std::uint64_t frontier = state.bank_frontier(projection.coordinate);
  std::size_t selected = ring.size();
  bool hit = false;
  if (projection.coordinate.ordinal == frontier) {
    selected = find_hit(frames, validated.region, ring, validated.demand.key);
    hit = selected != ring.size();
  } else if (future_hit_exists(frames, validated.region, ring,
                               validated.demand.key)) {
    return false;
  }
  if (selected == ring.size()) {
    if (projection.coordinate.ordinal != frontier &&
        !future_capacity_available(state, frames, projection, validated.region,
                                   ring, frontier)) {
      return false;
    }
    selected = find_empty(frames, validated.region, ring);
    if (selected == ring.size()) {
      selected = find_farthest_victim(state, frames, projection,
                                      validated.region, ring, frontier);
    }
  }
  if (selected == ring.size()) {
    return false;
  }
  const std::uint32_t target =
      validated.region.first + static_cast<std::uint32_t>(selected);
  const std::uint32_t reuse_frame =
      hit || !validated.reuse
          ? std::numeric_limits<std::uint32_t>::max()
          : find_reuse_frame(frames, validated.host_regions, active_banks,
                             validated.reuse.key, target);
  result = Selection{
      .slot = selected,
      .reuse_frame = reuse_frame == std::numeric_limits<std::uint32_t>::max()
                         ? 0u
                         : reuse_frame,
      .hit = hit,
      .reuses_frame = reuse_frame != std::numeric_limits<std::uint32_t>::max(),
  };
  return true;
}

} // namespace rund::compute::detail::residency::physical::fetch

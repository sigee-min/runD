#pragma once

#include "internal.hpp"

namespace rund::compute::detail::residency::physical::fetch {

template <typename State>
[[nodiscard]] bool
commit_state(State &state, const std::span<Authority::Frame> frames,
             const execution::SlidingProjection &projection,
             const std::span<PageUse> uses, const std::size_t use,
             const Projection &validated, const Selection selected,
             Commit &result) noexcept {
  auto ring = state.input_ring(projection.coordinate);
  auto &cell = ring[selected.slot];
  Authority::Frame &frame =
      frames[static_cast<std::size_t>(validated.region.first) + selected.slot];
  Authority::Frame *reuse_frame = nullptr;
  if (selected.reuses_frame) {
    if (selected.reuse_frame >= frames.size()) {
      return false;
    }
    reuse_frame = &frames[selected.reuse_frame];
    if (!validated.reuse || !reuse_frame->assigned ||
        reuse_frame->tier != FrameTier::Host ||
        reuse_frame->role != FrameRole::Input ||
        reuse_frame->state != FrameState::Resident ||
        !reuse_frame->dirty.empty() ||
        reuse_frame->key != validated.reuse.key) {
      return false;
    }
  }
  const std::uint64_t turn =
      cell.turn == std::numeric_limits<std::uint64_t>::max() ? 1u
                                                             : cell.turn + 1u;
  if ((selected.hit &&
       state.fetch_hits == std::numeric_limits<std::uint64_t>::max()) ||
      (!selected.hit &&
       state.fetch_calls == std::numeric_limits<std::uint64_t>::max())) {
    return false;
  }
  cell = typename State::InputCell{
      .coordinate = projection.coordinate,
      .key = uses[use].key,
      .pin = uses[use].pin,
      .turn = turn,
      .expected_bytes = validated.expected_bytes,
      .next_use = validated.demand.next_use,
      .physical_frame =
          validated.region.first + static_cast<std::uint32_t>(selected.slot),
      .use = static_cast<std::uint32_t>(use),
      .state = selected.hit ? execution::InputState::Ready
                            : execution::InputState::Fetching,
      .physical_handoff = true,
  };
  if (selected.hit) {
    ++state.fetch_hits;
    frame.next_use = validated.demand.next_use;
    frame.retain_until = validated.demand.retain_until;
  } else {
    ++state.fetch_calls;
    const Authority::Frame prior = frame;
    frame = Authority::Frame{.key = validated.demand.key,
                             .next_use = validated.demand.next_use,
                             .retain_until = validated.demand.retain_until,
                             .state = FrameState::Mapping,
                             .tier = prior.tier,
                             .role = prior.role,
                             .extent = prior.extent,
                             .view = prior.view,
                             .assigned = prior.assigned};
  }
  if (reuse_frame != nullptr) {
    reuse_frame->state = FrameState::Pinned;
  }
  result = Commit{.turn = turn,
                  .physical_frame = cell.physical_frame,
                  .reuse_frame = selected.reuse_frame,
                  .reuses_frame = selected.reuses_frame};
  return true;
}

} // namespace rund::compute::detail::residency::physical::fetch

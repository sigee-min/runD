#include "../../../registry.hpp"

#include "../../../execution/plan.hpp"
#include "../../frame.hpp"
#include "../../internal.hpp"

namespace rund::compute::detail::residency {

AuthorityFailure Authority::admit_direct_execution_locked(
    const execution::Plan &plan,
    registry_model::ExecutionSlot &candidate) noexcept {
  // Q=1 native execution addresses locals [0,K) directly. Authenticate that
  // exact physical layout before native submit; a cached permutation falls
  // back without inventing a relocation loop.
  execution::Node input{};
  execution::Node dispatch{};
  execution::Node output{};
  if (!plan.project(
          execution::NodeId{.epoch = 0u, .phase = execution::Phase::Input},
          input) ||
      !plan.project(
          execution::NodeId{.epoch = 0u, .phase = execution::Phase::Dispatch},
          dispatch) ||
      !plan.project(
          execution::NodeId{.epoch = 0u, .phase = execution::Phase::Output},
          output) ||
      input.input_count == 0u || input.input_count != dispatch.input_count ||
      dispatch.input_count != dispatch.output_count ||
      dispatch.output_count != output.output_count ||
      input.input_count > input.route.source.count ||
      input.input_count > input.route.target.count ||
      output.output_count > dispatch.route.target.count ||
      output.output_count > output.route.target.count) {
    return AuthorityFailure::Invalid;
  }
  const std::size_t count = input.input_count;
  const auto available = [this](const FrameRegion region, const CacheKey key,
                                const std::size_t local) noexcept {
    const std::size_t exact = static_cast<std::size_t>(region.first) + local;
    const std::size_t existing =
        find_frame(frames_, key, region.first, region.count);
    if (existing != frames_.size() && existing != exact) {
      return AuthorityFailure::Capacity;
    }
    const Frame &frame = frames_[exact];
    if (frame.state == FrameState::Mapping ||
        frame.state == FrameState::Pinned ||
        frame.state == FrameState::Writeback || !frame.dirty.empty()) {
      return AuthorityFailure::Busy;
    }
    return frame.state == FrameState::Empty ||
                   frame.state == FrameState::Resident
               ? AuthorityFailure::None
               : AuthorityFailure::Invalid;
  };
  for (std::size_t local = 0u; local < count; ++local) {
    const AuthorityFailure host =
        available(input.route.source, input.input[local].key, local);
    const AuthorityFailure device =
        available(input.route.target, input.input[local].key, local);
    const AuthorityFailure produced =
        available(dispatch.route.target, output.output[local].key, local);
    const AuthorityFailure drain =
        available(output.route.target, output.output[local].key, local);
    if (host != AuthorityFailure::None || device != AuthorityFailure::None ||
        produced != AuthorityFailure::None || drain != AuthorityFailure::None) {
      if (host == AuthorityFailure::Capacity ||
          device == AuthorityFailure::Capacity ||
          produced == AuthorityFailure::Capacity ||
          drain == AuthorityFailure::Capacity) {
        return AuthorityFailure::Capacity;
      }
      if (host == AuthorityFailure::Busy || device == AuthorityFailure::Busy ||
          produced == AuthorityFailure::Busy ||
          drain == AuthorityFailure::Busy) {
        return AuthorityFailure::Busy;
      }
      return AuthorityFailure::Invalid;
    }
  }

  const auto remember = [this, &candidate](const std::uint32_t index) {
    candidate.undo_frames[candidate.undo_count] = index;
    candidate.undo[candidate.undo_count] = frames_[index];
    ++candidate.undo_count;
  };
  const auto transition = [&candidate](const std::size_t service,
                                       const CacheTransition value) {
    const std::size_t index = candidate.service_transition_count[service]++;
    candidate.service_transitions[service][index] = value;
  };
  const auto admit_input = [this, &candidate, &remember, &transition](
                               const FrameRegion region, const CacheUse &use,
                               const std::size_t local, const bool backing) {
    const std::uint32_t frame =
        region.first + static_cast<std::uint32_t>(local);
    const Frame prior = frames_[frame];
    const bool hit =
        prior.state == FrameState::Resident && prior.key == use.key;
    remember(frame);
    if (!hit) {
      if (prior.state != FrameState::Empty) {
        transition(0u, CacheTransition{.key = prior.key,
                                       .frame = frame,
                                       .kind = TransitionKind::Unmap});
      }
      if (backing) {
        transition(0u, CacheTransition{.key = use.key,
                                       .frame = frame,
                                       .kind = TransitionKind::Fetch});
      }
      transition(0u, CacheTransition{.key = use.key,
                                     .frame = frame,
                                     .kind = TransitionKind::Map});
      frames_[frame] = Frame{.key = use.key,
                             .next_use = use.next_use,
                             .retain_until = use.retain_until,
                             .state = FrameState::Mapping,
                             .tier = prior.tier,
                             .role = prior.role,
                             .extent = prior.extent,
                             .view = prior.view,
                             .assigned = true};
    } else {
      frames_[frame].next_use = use.next_use;
      frames_[frame].retain_until = use.retain_until;
      frames_[frame].state = FrameState::Pinned;
    }
    const std::size_t binding = candidate.service_binding_count[0u]++;
    candidate.service_bindings[0u][binding] = CacheBinding{
        .key = use.key,
        .frame = frame,
        .access = Access::Read,
        .prior_dirty = prior.dirty,
        .next_use = use.next_use,
        .retain_until = use.retain_until,
        // `fetch` is local to this binding's tier. Host means backing supply;
        // Device means the authenticated Host-to-Device/coherent obligation.
        .fetch = !hit,
    };
    if (!hit) {
      const std::uint32_t bit = std::uint32_t{1u} << local;
      if (backing) {
        candidate.service_backing_mask[0u] |= bit;
      } else {
        candidate.service_transfer_mask[0u] |= bit;
      }
    }
  };
  for (std::size_t local = 0u; local < count; ++local) {
    admit_input(input.route.source, input.input[local], local, true);
  }
  for (std::size_t local = 0u; local < count; ++local) {
    admit_input(input.route.target, input.input[local], local, false);
  }
  // Save the exact prior output rows, but do not mark either valid before its
  // producer owns the bytes.
  for (std::size_t local = 0u; local < count; ++local) {
    remember(dispatch.route.target.first + static_cast<std::uint32_t>(local));
    remember(output.route.target.first + static_cast<std::uint32_t>(local));
  }
  candidate.cache_admitted = true;
  return AuthorityFailure::None;
}

} // namespace rund::compute::detail::residency

#include "../../registry.hpp"
#include "../internal.hpp"
#include "../lease_state.hpp"
#include "internal.hpp"

#include <algorithm>
#include <limits>
#include <new>

namespace rund::compute::detail::residency {

AuthorityResult
Authority::begin_transform(const std::span<const CacheUse> inputs,
                           const FrameRegion input_region,
                           const std::span<const CacheUse> outputs,
                           const FrameRegion output_region) noexcept {
  std::lock_guard lock{gate_};
  if (view_commit_quarantined_locked() || view_commit_inflight_locked() ||
      execution_state_.slot.token != 0u ||
      retry_ready(cycle_state_.graph_persists)) {
    return AuthorityResult{.failure = AuthorityFailure::Busy};
  }
  if (std::any_of(cycle_state_.epochs.begin(), cycle_state_.epochs.end(),
                  [](const LeaseSlot &slot) {
                    return slot.state == LeaseState::Prepared;
                  })) {
    return AuthorityResult{.failure = AuthorityFailure::Busy};
  }
  const auto valid_region = [this](const FrameRegion region) noexcept {
    const std::size_t first = region.first;
    const std::size_t count = region.count;
    return count != 0u && first <= frames_.size() &&
           count <= frames_.size() - first &&
           std::all_of(frames_.begin() + static_cast<std::ptrdiff_t>(first),
                       frames_.begin() +
                           static_cast<std::ptrdiff_t>(first + count),
                       [region](const Frame &frame) {
                         return frame.assigned && frame.tier == region.tier &&
                                frame.role == region.role;
                       });
  };
  const std::size_t input_end =
      static_cast<std::size_t>(input_region.first) + input_region.count;
  const std::size_t output_end =
      static_cast<std::size_t>(output_region.first) + output_region.count;
  const bool legal_roles = (input_region.role == FrameRole::Input &&
                            (output_region.role == FrameRole::Intermediate ||
                             output_region.role == FrameRole::Output)) ||
                           (input_region.role == FrameRole::Intermediate &&
                            output_region.role == FrameRole::Output);
  if (inputs.empty() || inputs.size() != outputs.size() ||
      inputs.size() > input_region.count ||
      outputs.size() > output_region.count || !legal_roles ||
      input_region.tier != output_region.tier ||
      input_region.count != output_region.count ||
      !(input_end <= output_region.first || output_end <= input_region.first) ||
      !valid_region(input_region) || !valid_region(output_region)) {
    return AuthorityResult{.failure =
                               inputs.size() > input_region.count ||
                                       outputs.size() > output_region.count
                                   ? AuthorityFailure::Capacity
                                   : AuthorityFailure::Invalid};
  }
  for (std::size_t left = 0u; left < inputs.size(); ++left) {
    const CacheUse &input = inputs[left];
    const CacheUse &output = outputs[left];
    if (input.key.backing == 0u || input.access != Access::Read ||
        !input.dirty.empty() || output.key.backing == 0u ||
        output.access != Access::Write || output.dirty.empty() ||
        ((input.epoch == NeverUse) != (input.retain_until == NeverUse)) ||
        ((output.epoch == NeverUse) != (output.retain_until == NeverUse)) ||
        input.epoch != output.epoch ||
        (left != 0u && input.epoch != inputs.front().epoch) ||
        (input.epoch != NeverUse && (input.retain_until < input.epoch ||
                                     output.retain_until < output.epoch)) ||
        output.dirty.bytes >
            std::numeric_limits<std::uint64_t>::max() - output.dirty.offset ||
        input.key.page != output.key.page) {
      return AuthorityResult{.failure = AuthorityFailure::Invalid};
    }
    for (std::size_t right = left + 1u; right < inputs.size(); ++right) {
      if (input.key == inputs[right].key || output.key == outputs[right].key) {
        return AuthorityResult{.failure = AuthorityFailure::Invalid};
      }
    }
  }
  const auto free = std::find_if(
      cycle_state_.epochs.begin(), cycle_state_.epochs.end(),
      [](const LeaseSlot &slot) { return slot.state == LeaseState::Free; });
  if (free == cycle_state_.epochs.end()) {
    return AuthorityResult{.failure = AuthorityFailure::Busy};
  }
  LeaseSlot &slot = *free;
  clear(slot);
  const auto fail = [&](const AuthorityFailure failure) noexcept {
    rollback(frames_, slot, false);
    clear(slot);
    return AuthorityResult{.failure = failure};
  };
  try {
    for (const CacheUse &use : inputs) {
      std::size_t frame =
          find_frame(frames_, use.key, input_region.first, input_region.count);
      if (frame != frames_.size() &&
          (frames_[frame].state == FrameState::Mapping ||
           frames_[frame].state == FrameState::Pinned ||
           frames_[frame].state == FrameState::Writeback)) {
        return fail(AuthorityFailure::Busy);
      }
      DirtyExtent prior_dirty{};
      bool fetch = false;
      if (frame == frames_.size()) {
        frame = lease_detail::select_frame(frames_, inputs, input_region.first,
                                           input_region.count, use.epoch);
        if (frame == frames_.size()) {
          return fail(AuthorityFailure::Capacity);
        }
        save(slot, static_cast<std::uint32_t>(frame), frames_[frame]);
        prior_dirty = frames_[frame].dirty;
        if (frames_[frame].state != FrameState::Empty) {
          if (!frames_[frame].dirty.empty()) {
            slot.transitions.push_back(CacheTransition{
                .key = frames_[frame].key,
                .frame = static_cast<std::uint32_t>(frame),
                .kind = lease_detail::retire_dirty(frames_[frame].key),
                .dirty = frames_[frame].dirty,
            });
          }
          slot.transitions.push_back(CacheTransition{
              .key = frames_[frame].key,
              .frame = static_cast<std::uint32_t>(frame),
              .kind = TransitionKind::Unmap,
          });
        }
        if (use.key.domain == CacheDomain::Transient) {
          return fail(AuthorityFailure::Invalid);
        }
        slot.transitions.push_back(CacheTransition{
            .key = use.key,
            .frame = static_cast<std::uint32_t>(frame),
            .kind = TransitionKind::Fetch,
        });
        slot.transitions.push_back(CacheTransition{
            .key = use.key,
            .frame = static_cast<std::uint32_t>(frame),
            .kind = TransitionKind::Map,
        });
        const FrameTier frame_tier = frames_[frame].tier;
        const FrameRole frame_role = frames_[frame].role;
        const std::uint64_t extent = frames_[frame].extent;
        const std::uint64_t view = frames_[frame].view;
        frames_[frame] = Frame{.key = use.key,
                               .next_use = use.next_use,
                               .retain_until = use.retain_until,
                               .state = FrameState::Mapping,
                               .tier = frame_tier,
                               .role = frame_role,
                               .extent = extent,
                               .view = view,
                               .assigned = true};
        fetch = true;
      } else {
        save(slot, static_cast<std::uint32_t>(frame), frames_[frame]);
        prior_dirty = frames_[frame].dirty;
        frames_[frame].next_use = use.next_use;
        frames_[frame].retain_until = use.retain_until;
        frames_[frame].state = FrameState::Pinned;
      }
      slot.bindings.push_back(
          CacheBinding{.key = use.key,
                       .frame = static_cast<std::uint32_t>(frame),
                       .access = use.access,
                       .prior_dirty = prior_dirty,
                       .fetch = fetch});
    }
    for (std::size_t index = 0u; index < outputs.size(); ++index) {
      const CacheUse &use = outputs[index];
      const CacheBinding &input = slot.bindings[index];
      const std::size_t local = input.frame - input_region.first;
      const std::size_t frame = output_region.first + local;
      const std::size_t existing = find_frame(
          frames_, use.key, output_region.first, output_region.count);
      if ((existing != frames_.size() && existing != frame) ||
          frames_[frame].state == FrameState::Mapping ||
          frames_[frame].state == FrameState::Pinned ||
          frames_[frame].state == FrameState::Writeback ||
          !frames_[frame].dirty.empty() ||
          (use.epoch != NeverUse && frames_[frame].retain_until != NeverUse &&
           use.epoch <= frames_[frame].retain_until)) {
        return fail(existing != frames_.size() && existing != frame
                        ? AuthorityFailure::Capacity
                        : AuthorityFailure::Busy);
      }
      const DirtyExtent prior_dirty = frames_[frame].dirty;
      save(slot, static_cast<std::uint32_t>(frame), frames_[frame]);
      if (frames_[frame].key != use.key ||
          frames_[frame].state == FrameState::Empty) {
        if (frames_[frame].state != FrameState::Empty) {
          slot.transitions.push_back(CacheTransition{
              .key = frames_[frame].key,
              .frame = static_cast<std::uint32_t>(frame),
              .kind = TransitionKind::Unmap,
          });
        }
        slot.transitions.push_back(CacheTransition{
            .key = use.key,
            .frame = static_cast<std::uint32_t>(frame),
            .kind = TransitionKind::Map,
        });
        const FrameTier frame_tier = frames_[frame].tier;
        const FrameRole frame_role = frames_[frame].role;
        const std::uint64_t extent = frames_[frame].extent;
        const std::uint64_t view = frames_[frame].view;
        frames_[frame] = Frame{.key = use.key,
                               .next_use = use.next_use,
                               .retain_until = use.retain_until,
                               .state = FrameState::Mapping,
                               .tier = frame_tier,
                               .role = frame_role,
                               .extent = extent,
                               .view = view,
                               .assigned = true};
      } else {
        frames_[frame].next_use = use.next_use;
        frames_[frame].retain_until = use.retain_until;
        frames_[frame].state = FrameState::Pinned;
      }
      slot.bindings.push_back(CacheBinding{
          .key = use.key,
          .frame = static_cast<std::uint32_t>(frame),
          .access = use.access,
          .dirty = use.dirty,
          .prior_dirty = prior_dirty,
      });
    }
  } catch (const std::bad_alloc &) {
    return fail(AuthorityFailure::Capacity);
  }
  slot.token = next_token(credentials_.next_token);
  if (slot.token == 0u) {
    return fail(AuthorityFailure::Capacity);
  }
  slot.generation = next_generation(credentials_.next_generation);
  if (slot.generation == 0u) {
    return fail(AuthorityFailure::Capacity);
  }
  slot.state = LeaseState::Prepared;
  return AuthorityResult{.failure = AuthorityFailure::None,
                         .lease = EpochLease{
                             .bindings = slot.bindings,
                             .transitions = slot.transitions,
                             .token = slot.token,
                             .generation = slot.generation,
                         }};
}

} // namespace rund::compute::detail::residency

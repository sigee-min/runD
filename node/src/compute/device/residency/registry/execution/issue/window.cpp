#include "../../../registry/execution_owner.hpp"

#include "../../../execution/plan.hpp"
#include "../../../execution/sliding.hpp"
#include "../../frame.hpp"
#include "../../internal.hpp"
#include "../../lease_state.hpp"

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <limits>

namespace rund::compute::detail::residency {

bool ExecutionOwner::issue_execution_window_service_locked(
    const execution::Plan &plan, const execution::Node &node,
    const std::size_t bank) noexcept {
  const std::size_t service_index =
      node.id.phase == execution::Phase::Output ? 1u : 0u;
  const std::size_t count = node.id.phase == execution::Phase::Input
                                ? node.input_count
                                : node.output_count;
  if (count == 0u || count > execution::UseCapacity ||
      authority_.execution_state_.slot
              .window_service_epoch[service_index][bank] != NeverUse) {
    return false;
  }
  auto &bindings = authority_.execution_state_.slot
                       .window_service_bindings[service_index][bank];
  auto &transitions = authority_.execution_state_.slot
                          .window_service_transitions[service_index][bank];
  std::size_t binding_count = 0u;
  std::size_t transition_count = 0u;
  std::uint32_t backing_mask = 0u;
  std::uint32_t transfer_mask = 0u;
  std::uint32_t coherent_mask = 0u;
  const auto exact_available = [this](const FrameRegion region,
                                      const CacheKey key,
                                      const std::size_t local) noexcept {
    if (local >= region.count) {
      return false;
    }
    const std::size_t exact = static_cast<std::size_t>(region.first) + local;
    const std::size_t existing =
        find_frame(authority_.frames_, key, region.first, region.count);
    const ExecutionFrame &frame = authority_.frames_[exact];
    return (existing == authority_.frames_.size() || existing == exact) &&
           (frame.state == FrameState::Empty ||
            frame.state == FrameState::Resident) &&
           frame.dirty.empty();
  };
  const auto remember = [this](const std::uint32_t frame) noexcept {
    const auto end = authority_.execution_state_.slot.undo_frames.begin() +
                     static_cast<std::ptrdiff_t>(
                         authority_.execution_state_.slot.undo_count);
    if (std::find(authority_.execution_state_.slot.undo_frames.begin(), end,
                  frame) != end) {
      return true;
    }
    if (authority_.execution_state_.slot.undo_count ==
        authority_.execution_state_.slot.undo_frames.size()) {
      return false;
    }
    authority_.execution_state_.slot
        .undo_frames[authority_.execution_state_.slot.undo_count] = frame;
    authority_.execution_state_.slot
        .undo[authority_.execution_state_.slot.undo_count] =
        authority_.frames_[frame];
    ++authority_.execution_state_.slot.undo_count;
    return true;
  };
  if (node.id.phase == execution::Phase::Input) {
    const auto &host_regions = plan.host_input_regions();
    if (host_regions[0].tier != FrameTier::Host ||
        host_regions[0].role != FrameRole::Input ||
        host_regions[1].tier != FrameTier::Host ||
        host_regions[1].role != FrameRole::Input ||
        host_regions[0].first + host_regions[0].count !=
            host_regions[1].first) {
      return false;
    }
    const std::size_t host_first = host_regions[0].first;
    const std::size_t host_count =
        static_cast<std::size_t>(host_regions[0].count) + host_regions[1].count;
    std::array<std::uint32_t, execution::UseCapacity> host_frames{};
    std::array<std::uint32_t, execution::UseCapacity> device_duplicates{};
    std::array<bool, execution::UseCapacity> device_hits{};
    device_duplicates.fill(std::numeric_limits<std::uint32_t>::max());
    std::size_t host_frame_count = 0u;
    const auto chosen = [&host_frames,
                         &host_frame_count](const std::size_t frame) noexcept {
      return std::find(host_frames.begin(),
                       host_frames.begin() +
                           static_cast<std::ptrdiff_t>(host_frame_count),
                       frame) !=
             host_frames.begin() +
                 static_cast<std::ptrdiff_t>(host_frame_count);
    };
    const auto select_host = [this, &node, host_first, host_count,
                              &chosen](const CacheUse &use) noexcept {
      const std::size_t existing =
          find_frame(authority_.frames_, use.key, host_first, host_count);
      if (existing != authority_.frames_.size() && !chosen(existing) &&
          authority_.frames_[existing].state == FrameState::Resident &&
          authority_.frames_[existing].dirty.empty()) {
        return existing;
      }
      const std::size_t end = host_first + host_count;
      for (std::size_t frame = host_first; frame < end; ++frame) {
        if (!chosen(frame) && authority_.frames_[frame].assigned &&
            authority_.frames_[frame].state == FrameState::Empty) {
          return frame;
        }
      }
      // After the same-bank native terminal, a page earlier than this
      // epoch's first page has already been consumed in the current
      // invocation. Replace the latest such page; never evict a future page
      // merely to stage a one-shot miss. If none is safe, coherent backing
      // fill goes directly to the exact Device target below.
      const std::uint64_t first_page = node.input[0u].key.page;
      std::size_t selected = authority_.frames_.size();
      for (std::size_t frame = host_first; frame < end; ++frame) {
        const ExecutionFrame &candidate = authority_.frames_[frame];
        if (chosen(frame) || !candidate.assigned ||
            candidate.state != FrameState::Resident ||
            !candidate.dirty.empty() ||
            candidate.key.backing != use.key.backing ||
            candidate.key.version != use.key.version ||
            candidate.key.page >= first_page ||
            demanded(
                std::span<const CacheUse>{node.input.data(), node.input_count},
                candidate.key)) {
          continue;
        }
        if (selected == authority_.frames_.size() ||
            authority_.frames_[selected].key.page < candidate.key.page) {
          selected = frame;
        }
      }
      return selected;
    };
    for (std::size_t local = 0u; local < count; ++local) {
      const std::size_t host = select_host(node.input[local]);
      const std::uint32_t target =
          node.route.target.first + static_cast<std::uint32_t>(local);
      const std::size_t duplicate =
          find_frame(authority_.frames_, node.input[local].key,
                     node.route.target.first, node.route.target.count);
      const ExecutionFrame &target_frame = authority_.frames_[target];
      device_hits[local] = target_frame.state == FrameState::Resident &&
                           target_frame.key == node.input[local].key;
      const bool target_available =
          (target_frame.state == FrameState::Empty ||
           target_frame.state == FrameState::Resident) &&
          target_frame.dirty.empty();
      const bool duplicate_available =
          duplicate == authority_.frames_.size() || duplicate == target ||
          (authority_.frames_[duplicate].state == FrameState::Resident &&
           authority_.frames_[duplicate].dirty.empty());
      if (!target_available || !duplicate_available ||
          (host != authority_.frames_.size() &&
           !remember(static_cast<std::uint32_t>(host))) ||
          !remember(target) ||
          (duplicate != authority_.frames_.size() && duplicate != target &&
           !remember(static_cast<std::uint32_t>(duplicate)))) {
        return false;
      }
      if (duplicate != authority_.frames_.size() && duplicate != target) {
        device_duplicates[local] = static_cast<std::uint32_t>(duplicate);
      }
      host_frames[host_frame_count++] =
          host == authority_.frames_.size()
              ? std::numeric_limits<std::uint32_t>::max()
              : static_cast<std::uint32_t>(host);
    }
    const auto admit = [this, &bindings, &transitions, &binding_count,
                        &transition_count, &backing_mask, &transfer_mask](
                           const std::uint32_t frame, const CacheUse &use,
                           const std::size_t local, const bool backing) {
      const ExecutionFrame prior = authority_.frames_[frame];
      const bool hit =
          prior.state == FrameState::Resident && prior.key == use.key;
      if (!hit && prior.state != FrameState::Empty) {
        transitions[transition_count++] = CacheTransition{
            .key = prior.key, .frame = frame, .kind = TransitionKind::Unmap};
      }
      if (!hit && backing) {
        transitions[transition_count++] = CacheTransition{
            .key = use.key, .frame = frame, .kind = TransitionKind::Fetch};
      }
      if (!hit) {
        transitions[transition_count++] = CacheTransition{
            .key = use.key, .frame = frame, .kind = TransitionKind::Map};
        authority_.frames_[frame] =
            ExecutionFrame{.key = use.key,
                           .next_use = use.next_use,
                           .retain_until = use.retain_until,
                           .state = FrameState::Mapping,
                           .tier = prior.tier,
                           .role = prior.role,
                           .extent = prior.extent,
                           .view = prior.view,
                           .assigned = true};
      } else {
        authority_.frames_[frame].next_use = use.next_use;
        authority_.frames_[frame].retain_until = use.retain_until;
        authority_.frames_[frame].state = FrameState::Pinned;
      }
      bindings[binding_count++] = CacheBinding{
          .key = use.key,
          .frame = frame,
          .access = Access::Read,
          .prior_dirty = prior.dirty,
          .next_use = use.next_use,
          .retain_until = use.retain_until,
          .fetch = !hit,
      };
      if (!hit) {
        const std::uint32_t bit = std::uint32_t{1u} << local;
        (backing ? backing_mask : transfer_mask) |= bit;
      }
    };
    for (std::size_t local = 0u; local < count; ++local) {
      if (host_frames[local] == std::numeric_limits<std::uint32_t>::max()) {
        const bool fetch = !device_hits[local];
        bindings[binding_count++] = CacheBinding{
            .key = node.input[local].key,
            .frame = host_frames[local],
            .access = Access::Read,
            .next_use = node.input[local].next_use,
            .retain_until = node.input[local].retain_until,
            .fetch = fetch,
        };
        const std::uint32_t bit = std::uint32_t{1u} << local;
        coherent_mask |= bit;
        if (fetch) {
          backing_mask |= bit;
        }
      } else {
        admit(host_frames[local], node.input[local], local, true);
      }
    }
    for (std::size_t local = 0u; local < count; ++local) {
      if (device_duplicates[local] !=
          std::numeric_limits<std::uint32_t>::max()) {
        authority_.frames_[device_duplicates[local]] =
            frame_detail::empty(authority_.frames_[device_duplicates[local]]);
      }
    }
    for (std::size_t local = 0u; local < count; ++local) {
      admit(node.route.target.first + static_cast<std::uint32_t>(local),
            node.input[local], local, false);
    }
    transfer_mask &= ~coherent_mask;
  } else {
    for (std::size_t local = 0u; local < count; ++local) {
      const CacheUse &use = node.output[local];
      const std::uint32_t source =
          node.route.source.first + static_cast<std::uint32_t>(local);
      if (source >= authority_.frames_.size() ||
          authority_.frames_[source].key != use.key ||
          authority_.frames_[source].state != FrameState::Pinned ||
          authority_.frames_[source].dirty != use.dirty ||
          !exact_available(node.route.target, use.key, local)) {
        return false;
      }
    }
    for (std::size_t local = 0u; local < count; ++local) {
      const CacheUse &use = node.output[local];
      const std::uint32_t source =
          node.route.source.first + static_cast<std::uint32_t>(local);
      const std::uint32_t target =
          node.route.target.first + static_cast<std::uint32_t>(local);
      const ExecutionFrame source_frame = authority_.frames_[source];
      const ExecutionFrame prior = authority_.frames_[target];
      bindings[binding_count++] = CacheBinding{
          .key = use.key,
          .frame = source,
          .access = Access::Write,
          .dirty = use.dirty,
          .prior_dirty = source_frame.dirty,
          .next_use = use.next_use,
          .retain_until = use.retain_until,
      };
      if (prior.state != FrameState::Empty) {
        transitions[transition_count++] = CacheTransition{
            .key = prior.key,
            .frame = target,
            .kind = TransitionKind::Unmap,
        };
      }
      transitions[transition_count++] = CacheTransition{
          .key = use.key, .frame = target, .kind = TransitionKind::Map};
      transitions[transition_count++] = CacheTransition{
          .key = use.key,
          .frame = target,
          .kind = TransitionKind::Writeback,
          .dirty = use.dirty,
      };
      authority_.frames_[target] =
          ExecutionFrame{.key = use.key,
                         .next_use = use.next_use,
                         .retain_until = use.retain_until,
                         .state = FrameState::Mapping,
                         .tier = prior.tier,
                         .role = prior.role,
                         .extent = prior.extent,
                         .view = prior.view,
                         .assigned = true};
      bindings[binding_count++] = CacheBinding{
          .key = use.key,
          .frame = target,
          .access = Access::Write,
          .dirty = use.dirty,
          .prior_dirty = prior.dirty,
          .next_use = use.next_use,
          .retain_until = use.retain_until,
      };
      const std::uint32_t bit = std::uint32_t{1u} << local;
      backing_mask |= bit;
      transfer_mask |= bit;
    }
  }
  authority_.execution_state_.slot
      .window_service_binding_count[service_index][bank] = binding_count;
  authority_.execution_state_.slot
      .window_service_transition_count[service_index][bank] = transition_count;
  authority_.execution_state_.slot
      .window_service_backing_mask[service_index][bank] = backing_mask;
  authority_.execution_state_.slot
      .window_service_transfer_mask[service_index][bank] = transfer_mask;
  authority_.execution_state_.slot
      .window_service_coherent_mask[service_index][bank] = coherent_mask;
  authority_.execution_state_.slot.window_service_epoch[service_index][bank] =
      node.id.epoch;
  return true;
}

} // namespace rund::compute::detail::residency

#include "../../registry/frame.hpp"
#include "../../registry/internal.hpp"
#include "internal.hpp"

namespace rund::compute::detail::residency {
namespace {

bool same_identity(const Authority::Frame &left,
                   const Authority::Frame &right) noexcept {
  return left.assigned && right.assigned && left.key == right.key &&
         left.extent == right.extent && left.view == right.view;
}

const Authority::Frame *prior_for(const Authority::LeaseSlot &epoch,
                                  const std::uint32_t frame) noexcept {
  for (std::size_t index = 0u; index < epoch.undo_frames.size(); ++index) {
    if (epoch.undo_frames[index] == frame) {
      return &epoch.undo[index];
    }
  }
  return nullptr;
}

bool read_alias_safe(const std::vector<Authority::Frame> &frames,
                     const Authority::LeaseSlot &epoch,
                     const std::uint32_t frame,
                     const CacheBinding *const binding) noexcept {
  if (binding != nullptr &&
      (binding->access != Access::Read || binding->fetch ||
       binding->relocated || binding->retire_on_success ||
       !binding->dirty.empty() || !binding->prior_dirty.empty())) {
    return false;
  }
  for (const CacheBinding &candidate : epoch.bindings) {
    if (candidate.frame == frame &&
        (candidate.access != Access::Read || candidate.fetch ||
         candidate.relocated || candidate.retire_on_success ||
         !candidate.dirty.empty() || !candidate.prior_dirty.empty())) {
      return false;
    }
  }
  if (std::find(epoch.relocation_frames.begin(), epoch.relocation_frames.end(),
                frame) != epoch.relocation_frames.end() ||
      frame >= frames.size()) {
    return false;
  }
  const Authority::Frame *const prior = prior_for(epoch, frame);
  const Authority::Frame &current = frames[frame];
  return prior != nullptr && prior->state == FrameState::Resident &&
         prior->dirty.empty() && same_identity(*prior, current) &&
         current.dirty.empty() &&
         (current.state == FrameState::Resident ||
          current.state == FrameState::Pinned ||
          current.state == FrameState::Mapping);
}

bool valid_source(const std::vector<Authority::Frame> &frames,
                  const Authority::LeaseSlot &epoch) noexcept {
  if (epoch.state != Authority::LeaseState::Computing || epoch.cycle != 0u ||
      epoch.cpu_key || epoch.cpu_bound ||
      epoch.undo_frames.size() != epoch.undo.size()) {
    return false;
  }
  for (std::size_t index = 0u; index < epoch.undo_frames.size(); ++index) {
    const std::uint32_t frame = epoch.undo_frames[index];
    if (frame >= frames.size() ||
        (frames[frame].alias_claims != 0u &&
         !read_alias_safe(frames, epoch, frame, nullptr))) {
      return false;
    }
  }
  for (std::size_t index = 0u; index < epoch.bindings.size(); ++index) {
    const CacheBinding &binding = epoch.bindings[index];
    if (binding.frame >= frames.size() ||
        (frames[binding.frame].alias_claims != 0u &&
         !read_alias_safe(frames, epoch, binding.frame, &binding))) {
      return false;
    }
    for (std::size_t prior = 0u; prior < index; ++prior) {
      if (epoch.bindings[prior].frame == binding.frame) {
        return false;
      }
    }
    const Authority::Frame &frame = frames[binding.frame];
    if (frame.key != binding.key ||
        (frame.state != FrameState::Pinned &&
         frame.state != FrameState::Mapping) ||
        (binding.retire_on_success && binding.access != Access::Read) ||
        (binding.retire_on_success &&
         binding.key.domain != CacheDomain::Transient) ||
        (binding.retire_on_success && frame.dirty.empty())) {
      return false;
    }
  }
  return true;
}

void apply_source(std::vector<Authority::Frame> &frames,
                  Authority::LeaseSlot &epoch, const bool success,
                  const bool invalidate) noexcept {
  if (!success) {
    for (std::size_t index = 0u; index < epoch.undo.size(); ++index) {
      frames[epoch.undo_frames[index]] = epoch.undo[index];
    }
    for (const CacheBinding &binding : epoch.bindings) {
      if (binding.fetch ||
          (invalidate && (epoch.ports.empty() || writes(binding.access)))) {
        frames[binding.frame] = frame_detail::empty(frames[binding.frame]);
      }
    }
    for (const std::uint32_t frame : epoch.relocation_frames) {
      frames[frame] = frame_detail::empty(frames[frame]);
    }
    ::rund::compute::detail::residency::clear(epoch);
    return;
  }
  for (const CacheBinding &binding : epoch.bindings) {
    Authority::Frame &frame = frames[binding.frame];
    if (binding.retire_on_success) {
      frame = frame_detail::empty(frame);
      continue;
    }
    if (binding.access == Access::Write) {
      frame.dirty = binding.dirty;
    }
    frame.state =
        frame.dirty.empty() ? FrameState::Resident : FrameState::Dirty;
  }
  ::rund::compute::detail::residency::clear(epoch);
}

bool valid_destination(const std::vector<Authority::Frame> &frames,
                       const Authority::LeaseSlot &destination) noexcept {
  if (destination.state != Authority::LeaseState::Prepared ||
      destination.cycle != 0u || destination.bindings.empty() ||
      destination.undo_frames.size() != destination.undo.size()) {
    return false;
  }
  for (const std::uint32_t frame : destination.relocation_frames) {
    if (frame >= frames.size() || !frames[frame].assigned ||
        (frames[frame].state != FrameState::Pinned &&
         frames[frame].state != FrameState::Mapping)) {
      return false;
    }
  }
  for (std::size_t index = 0u; index < destination.bindings.size(); ++index) {
    const CacheBinding &binding = destination.bindings[index];
    if (binding.frame >= frames.size() ||
        (frames[binding.frame].state != FrameState::Mapping &&
         frames[binding.frame].state != FrameState::Pinned) ||
        (!binding.relocated && frames[binding.frame].key != binding.key)) {
      return false;
    }
    for (std::size_t prior = 0u; prior < index; ++prior) {
      if (destination.bindings[prior].frame == binding.frame) {
        return false;
      }
    }
  }
  return true;
}

void apply_destination(std::vector<Authority::Frame> &frames,
                       Authority::LeaseSlot &destination) noexcept {
  for (const std::uint32_t frame_index : destination.relocation_frames) {
    const bool target =
        std::any_of(destination.bindings.begin(), destination.bindings.end(),
                    [frame_index](const CacheBinding &binding) {
                      return binding.frame == frame_index;
                    });
    if (!target) {
      frames[frame_index] = frame_detail::empty(frames[frame_index]);
    }
  }
  for (const CacheBinding &binding : destination.bindings) {
    Authority::Frame &frame = frames[binding.frame];
    if (binding.relocated) {
      const FrameTier tier = frame.tier;
      const FrameRole role = frame.role;
      const bool assigned = frame.assigned;
      const std::uint64_t extent = frame.extent;
      const std::uint64_t view = frame.view;
      frame = Authority::Frame{.key = binding.key,
                               .next_use = binding.next_use,
                               .retain_until = binding.retain_until,
                               .state = FrameState::Pinned,
                               .tier = tier,
                               .role = role,
                               .dirty = binding.prior_dirty,
                               .extent = extent,
                               .view = view,
                               .assigned = assigned};
    } else {
      frame.state = FrameState::Pinned;
    }
  }
  destination.state = Authority::LeaseState::Computing;
}

bool valid_abort(const std::vector<Authority::Frame> &frames,
                 const Authority::LeaseSlot &epoch) noexcept {
  if (epoch.state != Authority::LeaseState::Prepared || epoch.cycle != 0u ||
      epoch.cpu_key || epoch.cpu_bound ||
      epoch.undo_frames.size() != epoch.undo.size()) {
    return false;
  }
  for (const std::uint32_t frame : epoch.undo_frames) {
    if (frame >= frames.size() || frames[frame].alias_claims != 0u) {
      return false;
    }
  }
  for (std::size_t index = 0u; index < epoch.bindings.size(); ++index) {
    const CacheBinding &binding = epoch.bindings[index];
    if (binding.frame >= frames.size() ||
        frames[binding.frame].alias_claims != 0u) {
      return false;
    }
    for (std::size_t prior = 0u; prior < index; ++prior) {
      if (epoch.bindings[prior].frame == binding.frame) {
        return false;
      }
    }
  }
  for (const std::uint32_t frame : epoch.relocation_frames) {
    if (frame >= frames.size() || frames[frame].alias_claims != 0u) {
      return false;
    }
  }
  return true;
}

void apply_abort(std::vector<Authority::Frame> &frames,
                 Authority::LeaseSlot &epoch, const bool invalidate) noexcept {
  for (std::size_t index = 0u; index < epoch.undo.size(); ++index) {
    frames[epoch.undo_frames[index]] = epoch.undo[index];
  }
  for (const CacheBinding &binding : epoch.bindings) {
    if (binding.fetch ||
        (invalidate && (epoch.ports.empty() || writes(binding.access)))) {
      frames[binding.frame] = frame_detail::empty(frames[binding.frame]);
    }
  }
  for (const std::uint32_t frame : epoch.relocation_frames) {
    frames[frame] = frame_detail::empty(frames[frame]);
  }
  ::rund::compute::detail::residency::clear(epoch);
}

} // namespace

bool GraphPromoteOwner::release_graph_promote(
    execution::GraphPromote &&ticket) noexcept {
  if (!ticket || ticket.owner_ != &authority_ || !ticket.terminalled_) {
    return false;
  }
  const bool success = static_cast<bool>(ticket.completion_) &&
                       ticket.terminal_ == execution::TerminalKind::Known &&
                       !ticket.completion_may_write_;
  std::lock_guard lock{authority_.gate_};
  if (authority_.view_commit_quarantined_locked() ||
      ticket.source_count_ == 0u ||
      ticket.source_count_ > execution::GraphPromoteSourceCapacity ||
      ticket.destination_token_ == 0u ||
      ticket.page_count_ > execution::GraphPromoteCapacity) {
    return false;
  }
  const auto destination =
      std::find_if(authority_.cycle_state_.epochs.begin(),
                   authority_.cycle_state_.epochs.end(),
                   [&](const registry_model::LeaseSlot &slot) {
                     return slot.token == ticket.destination_token_;
                   });
  if (destination == authority_.cycle_state_.epochs.end() ||
      !valid_destination(authority_.frames_, *destination)) {
    return false;
  }
  std::array<registry_model::LeaseSlot *, execution::GraphPromoteSourceCapacity>
      sources{};
  for (std::size_t index = 0u; index < ticket.source_count_; ++index) {
    const std::uint64_t token = ticket.source_tokens_[index];
    if (token == 0u) {
      continue;
    }
    const auto source =
        std::find_if(authority_.cycle_state_.epochs.begin(),
                     authority_.cycle_state_.epochs.end(),
                     [token](const registry_model::LeaseSlot &slot) {
                       return slot.token == token;
                     });
    if (source == authority_.cycle_state_.epochs.end() ||
        source == destination || !valid_source(authority_.frames_, *source) ||
        std::find(sources.begin(), sources.begin() + index, &*source) !=
            sources.begin() + index) {
      return false;
    }
    sources[index] = &*source;
  }
  if (!success && !valid_abort(authority_.frames_, *destination)) {
    return false;
  }
  for (std::size_t index = 0u; index < ticket.source_count_; ++index) {
    if (sources[index] != nullptr) {
      // Forecast sources are read-only Host rows.  A failed H2D destination
      // must not roll those verified source rows back with the destination.
      apply_source(authority_.frames_, *sources[index], true, false);
    }
  }
  if (success) {
    apply_destination(authority_.frames_, *destination);
  } else {
    apply_abort(authority_.frames_, *destination, ticket.completion_may_write_);
  }
  ticket.clear();
  return true;
}

} // namespace rund::compute::detail::residency

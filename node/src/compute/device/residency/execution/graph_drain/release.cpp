#include "../../registry/frame.hpp"
#include "../../registry/graph_drain_owner.hpp"
#include "../../registry/internal.hpp"
#include "internal.hpp"

namespace rund::compute::detail::residency {
namespace {

bool valid_failure_source(const std::vector<Authority::Frame> &frames,
                          const Authority::LeaseSlot &source) noexcept {
  if (source.state != Authority::LeaseState::Drain ||
      source.undo_frames.size() != source.undo.size() ||
      source.transitions.size() != source.undo.size() ||
      source.transitions.empty()) {
    return false;
  }
  for (std::size_t index = 0u; index < source.transitions.size(); ++index) {
    const CacheTransition &transition = source.transitions[index];
    const std::uint32_t frame = source.undo_frames[index];
    if (transition.kind != TransitionKind::Migrate ||
        transition.frame != frame || frame >= frames.size() ||
        frames[frame].key != transition.key ||
        frames[frame].state != FrameState::Writeback ||
        frames[frame].dirty != transition.dirty) {
      return false;
    }
    for (std::size_t prior = 0u; prior < index; ++prior) {
      if (source.undo_frames[prior] == frame) {
        return false;
      }
    }
  }
  return true;
}

bool valid_failure_destination(
    const std::vector<Authority::Frame> &frames,
    const Authority::LeaseSlot &destination) noexcept {
  if ((destination.state != Authority::LeaseState::Prepared &&
       destination.state != Authority::LeaseState::Computing) ||
      destination.cycle != 0u || destination.cpu_key || destination.cpu_bound ||
      !destination.ports.empty() ||
      destination.undo_frames.size() != destination.undo.size()) {
    return false;
  }
  for (const std::uint32_t frame : destination.undo_frames) {
    if (frame >= frames.size() || frames[frame].alias_claims != 0u) {
      return false;
    }
  }
  for (std::size_t index = 0u; index < destination.bindings.size(); ++index) {
    const CacheBinding &binding = destination.bindings[index];
    if (binding.frame >= frames.size() || binding.access != Access::Write ||
        frames[binding.frame].alias_claims != 0u) {
      return false;
    }
    for (std::size_t prior = 0u; prior < index; ++prior) {
      if (destination.bindings[prior].frame == binding.frame) {
        return false;
      }
    }
  }
  for (const std::uint32_t frame : destination.relocation_frames) {
    if (frame >= frames.size() || frames[frame].alias_claims != 0u) {
      return false;
    }
  }
  return true;
}

bool valid_migration(const std::vector<Authority::Frame> &frames,
                     const Authority::LeaseSlot &source,
                     const Authority::LeaseSlot &destination) noexcept {
  if (source.state != Authority::LeaseState::Drain ||
      destination.state != Authority::LeaseState::Computing ||
      destination.bindings.empty() ||
      source.transitions.size() != destination.bindings.size() ||
      destination.undo_frames.size() != destination.bindings.size() ||
      destination.undo.size() != destination.bindings.size()) {
    return false;
  }
  for (const CacheTransition &transition : destination.transitions) {
    if (transition.kind != TransitionKind::Map &&
        transition.kind != TransitionKind::Unmap) {
      return false;
    }
  }
  for (std::size_t index = 0u; index < source.transitions.size(); ++index) {
    const CacheTransition transition = source.transitions[index];
    const CacheBinding binding = destination.bindings[index];
    const Authority::Frame &prior = destination.undo[index];
    if (transition.kind != TransitionKind::Migrate ||
        transition.frame >= frames.size() || binding.frame >= frames.size() ||
        transition.frame == binding.frame ||
        destination.undo_frames[index] != binding.frame ||
        !prior.dirty.empty() ||
        (prior.state != FrameState::Empty &&
         prior.state != FrameState::Resident)) {
      return false;
    }
    const Authority::Frame &source_frame = frames[transition.frame];
    const Authority::Frame &target = frames[binding.frame];
    if (!source_frame.assigned || source_frame.tier != FrameTier::Device ||
        source_frame.role != FrameRole::Output ||
        source_frame.key != transition.key ||
        source_frame.state != FrameState::Writeback ||
        source_frame.dirty.empty() || source_frame.dirty != transition.dirty ||
        source_frame.dirty != binding.dirty || !target.assigned ||
        target.tier != FrameTier::Host || target.role != FrameRole::Output ||
        target.key != binding.key ||
        (source_frame.transaction_generation == 0u) !=
            (target.transaction_generation == 0u) ||
        (source_frame.transaction_generation != 0u &&
         source_frame.transaction_generation !=
             target.transaction_generation) ||
        (target.state != FrameState::Pinned &&
         target.state != FrameState::Mapping) ||
        binding.access != Access::Write || source_frame.key != target.key) {
      return false;
    }
    for (std::size_t prior_index = 0u; prior_index < index; ++prior_index) {
      if (source.undo_frames[prior_index] == transition.frame ||
          destination.bindings[prior_index].frame == binding.frame) {
        return false;
      }
    }
  }
  return true;
}

void apply_migration(std::vector<Authority::Frame> &frames,
                     Authority::LeaseSlot &source,
                     Authority::LeaseSlot &destination) noexcept {
  for (const CacheTransition &transition : source.transitions) {
    frames[transition.frame] = frame_detail::empty(frames[transition.frame]);
  }
  for (const CacheBinding &binding : destination.bindings) {
    Authority::Frame &frame = frames[binding.frame];
    frame.dirty = binding.dirty;
    frame.state = FrameState::Dirty;
  }
  ::rund::compute::detail::residency::clear(source);
  ::rund::compute::detail::residency::clear(destination);
}

void apply_failure(std::vector<Authority::Frame> &frames,
                   Authority::LeaseSlot &source,
                   Authority::LeaseSlot &destination,
                   const bool invalidate) noexcept {
  for (std::size_t index = 0u; index < source.undo.size(); ++index) {
    frames[source.undo_frames[index]] = source.undo[index];
  }
  if (invalidate) {
    for (const CacheTransition &transition : source.transitions) {
      frames[transition.frame] = frame_detail::empty(frames[transition.frame]);
    }
  }
  for (std::size_t index = 0u; index < destination.undo.size(); ++index) {
    frames[destination.undo_frames[index]] = destination.undo[index];
  }
  for (const CacheBinding &binding : destination.bindings) {
    if (binding.fetch ||
        (invalidate && (destination.ports.empty() || writes(binding.access)))) {
      frames[binding.frame] = frame_detail::empty(frames[binding.frame]);
    }
  }
  for (const std::uint32_t frame : destination.relocation_frames) {
    frames[frame] = frame_detail::empty(frames[frame]);
  }
  ::rund::compute::detail::residency::clear(source);
  ::rund::compute::detail::residency::clear(destination);
}

} // namespace

bool GraphDrainOwner::release_graph_drain(
    execution::GraphDrain &&ticket) noexcept {
  if (!ticket || ticket.owner_ != &authority_ || !ticket.terminalled_) {
    return false;
  }
  const bool success = static_cast<bool>(ticket.completion_) &&
                       ticket.terminal_ == execution::TerminalKind::Known &&
                       !ticket.completion_may_write_;
  std::lock_guard lock{authority_.gate_};
  if (authority_.view_commit_quarantined_locked() ||
      ticket.source_token_ == 0u || ticket.destination_token_ == 0u) {
    return false;
  }
  const auto destination =
      std::find_if(authority_.cycle_state_.epochs.begin(),
                   authority_.cycle_state_.epochs.end(),
                   [&](const Authority::LeaseSlot &slot) {
                     return slot.token == ticket.destination_token_;
                   });
  if (destination == authority_.cycle_state_.epochs.end() ||
      authority_.cycle_state_.writeback.token != ticket.source_token_) {
    return false;
  }
  if (success) {
    if (!valid_migration(authority_.frames_, authority_.cycle_state_.writeback,
                         *destination)) {
      return false;
    }
    apply_migration(authority_.frames_, authority_.cycle_state_.writeback,
                    *destination);
  } else {
    if (!valid_failure_source(authority_.frames_,
                              authority_.cycle_state_.writeback) ||
        !valid_failure_destination(authority_.frames_, *destination)) {
      return false;
    }
    apply_failure(authority_.frames_, authority_.cycle_state_.writeback,
                  *destination, ticket.completion_may_write_);
  }
  ticket.clear();
  return true;
}

} // namespace rund::compute::detail::residency

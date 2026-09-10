#pragma once

#include "../graph_persist.hpp"

#include "../../registry/internal.hpp"
#include "../../registry/frame.hpp"

#include <kernel/core/checked.hpp>

#include <algorithm>
#include <array>
#include <limits>

namespace rund::compute::detail::residency::graph_persist_detail {

struct Projection final {
  std::array<PageUse, TiledGraphPortCapacity * execution::GraphPersistCapacity>
      uses{};
  Epoch epoch{};
  std::size_t use_count{};
};

[[nodiscard]] inline bool
validate_projection(const std::shared_ptr<const ResidencyPlan> &owner,
                    const TiledGraphInvocation &invocation,
                    const std::uint64_t batch, const std::size_t stage,
                    const std::uint32_t resource,
                    const std::span<const PageUse> selected,
                    const GraphMaterialization materialization,
                    const FrameRegion region, Projection &result) noexcept {
  result = {};
  if (owner == nullptr || !owner->graph_tiled() || !owner->identity() ||
      !invocation.owned_by(owner->tiled_graph()) || selected.empty() ||
      selected.size() > execution::GraphPersistCapacity ||
      region.tier != FrameTier::Host || region.role != FrameRole::Output ||
      region.count == 0u || selected.size() > region.count ||
      materialization.resource != resource ||
      materialization.key.backing == 0u || materialization.key.page != 0u ||
      materialization.key.domain != CacheDomain::Backing ||
      materialization.page_count != invocation.page_count()) {
    return false;
  }
  const TiledGraphPlan &plan = owner->tiled_graph();
  if (stage >= plan.stages().size()) {
    return false;
  }
  const TiledGraphResource *const declared = plan.resource(resource);
  PageRun run{};
  if (declared == nullptr ||
      declared->kind != GraphResourceKind::ExternalOutput ||
      declared->persistence != ResourcePersistence::Backing ||
      declared->page_bytes != materialization.page_bytes ||
      !invocation.batch(batch, run) || run.page_count == 0u ||
      run.page_count > execution::GraphPersistCapacity) {
    return false;
  }
  const TiledGraphStage &node = plan.stages()[stage];
  if (node.ports.empty() || node.ports.size() > TiledGraphPortCapacity ||
      run.page_count > result.uses.size() / node.ports.size()) {
    return false;
  }
  const auto port = std::find_if(node.ports.begin(), node.ports.end(),
                                 [resource](const TiledGraphPort row) {
                                   return row.resource == resource &&
                                          row.access == Access::Write;
                                 });
  if (port == node.ports.end() ||
      std::find_if(std::next(port), node.ports.end(),
                   [resource](const TiledGraphPort row) {
                     return row.resource == resource &&
                            row.access == Access::Write;
                   }) != node.ports.end()) {
    return false;
  }
  result.use_count =
      static_cast<std::size_t>(run.page_count) * node.ports.size();
  if (!invocation.project(
          batch, stage,
          std::span<PageUse>{result.uses.data(), result.use_count},
          result.epoch)) {
    return false;
  }
  const std::size_t port_index =
      static_cast<std::size_t>(port - node.ports.begin());
  const std::size_t first =
      port_index * static_cast<std::size_t>(run.page_count);
  const auto exact = std::span<const PageUse>{
      result.uses.data() + static_cast<std::ptrdiff_t>(first),
      static_cast<std::size_t>(run.page_count)};
  return exact.size() == selected.size() &&
         std::equal(exact.begin(), exact.end(), selected.begin()) &&
         std::all_of(selected.begin(), selected.end(), [](const PageUse use) {
           return use.access == Access::Write && use.dirty.bytes != 0u;
         });
}

inline void rollback(std::vector<registry_model::Frame> &frames,
                     registry_model::LeaseSlot &slot) noexcept {
  for (std::size_t index = 0u; index < slot.undo.size(); ++index) {
    if (slot.undo_frames[index] < frames.size()) {
      frames[slot.undo_frames[index]] = slot.undo[index];
    }
  }
}

enum class UndoMode : std::uint8_t { Drain, Retry, Staged };

[[nodiscard]] inline bool
check_core(const std::vector<registry_model::Frame> &frames,
           const registry_model::LeaseSlot &slot, const FrameRegion region,
           const UndoMode mode = UndoMode::Drain) noexcept {
  if (region.count == 0u || region.tier != FrameTier::Host ||
      region.role != FrameRole::Output || region.first > frames.size() ||
      region.count > frames.size() - region.first ||
      slot.transitions.size() > execution::GraphPersistCapacity ||
      slot.undo_frames.size() > execution::GraphPersistCapacity ||
      slot.undo.size() > execution::GraphPersistCapacity ||
      slot.transitions.size() != slot.undo_frames.size() ||
      slot.transitions.size() != slot.undo.size()) {
    return false;
  }
  if (slot.transitions.empty()) {
    return true;
  }

  const auto same = [](const registry_model::Frame &left,
                       const registry_model::Frame &right) noexcept {
    return left.key == right.key && left.next_use == right.next_use &&
           left.retain_until == right.retain_until && left.tier == right.tier &&
           left.role == right.role && left.dirty == right.dirty &&
           left.extent == right.extent && left.view == right.view &&
           left.direct_registration == right.direct_registration &&
           left.assigned == right.assigned &&
           left.alias_claims == right.alias_claims;
  };
  const auto staged = [](const registry_model::Frame &frame,
                         const registry_model::Frame &prior) noexcept {
    return frame.state == FrameState::Empty && frame.key == CacheKey{} &&
           frame.next_use == 0u && frame.retain_until == NeverUse &&
           frame.dirty.empty() && frame.extent == prior.extent &&
           frame.view == prior.view && frame.tier == prior.tier &&
           frame.role == prior.role && frame.assigned == prior.assigned &&
           frame.direct_registration == nullptr && frame.alias_claims == 0u;
  };

  for (std::size_t index = 0u; index < slot.transitions.size(); ++index) {
    const CacheTransition &transition = slot.transitions[index];
    const std::uint32_t frame_index = slot.undo_frames[index];
    if (transition.kind != TransitionKind::Writeback ||
        transition.dirty.bytes == 0u || frame_index != transition.frame ||
        frame_index < region.first ||
        frame_index >= region.first + region.count ||
        frame_index >= frames.size()) {
      return false;
    }
    for (std::size_t prior = 0u; prior < index; ++prior) {
      if (slot.undo_frames[prior] == frame_index) {
        return false;
      }
    }
    const registry_model::Frame &current = frames[frame_index];
    const registry_model::Frame &prior = slot.undo[index];
    if (current.tier != FrameTier::Host || current.role != FrameRole::Output ||
        prior.tier != FrameTier::Host || prior.role != FrameRole::Output ||
        !prior.assigned || prior.state != FrameState::Dirty ||
        prior.key != transition.key || prior.dirty != transition.dirty ||
        prior.alias_claims != 0u || current.alias_claims != 0u ||
        (mode != UndoMode::Staged && (transition.key != current.key ||
                                      transition.dirty != current.dirty))) {
      return false;
    }
    if (mode == UndoMode::Staged) {
      if (!staged(current, prior)) {
        return false;
      }
      continue;
    }
    if (mode == UndoMode::Drain) {
      if (current.state != FrameState::Writeback || !same(current, prior)) {
        return false;
      }
    } else if (current.state != FrameState::Dirty || !same(current, prior)) {
      return false;
    }
  }
  return true;
}

[[nodiscard]] inline bool
check_undo(const std::vector<registry_model::Frame> &frames,
           const registry_model::LeaseSlot &slot,
           const UndoMode mode = UndoMode::Drain) noexcept {
  const bool direct = slot.book_domain == 0u && slot.plan != Identity{} &&
                      slot.identity == GraphPersistIdentity{};
  const bool cpu = slot.book_domain != 0u && slot.identity.valid() &&
                   slot.identity.plan == slot.plan;
  return (direct || cpu) && check_core(frames, slot, slot.retry_region, mode);
}

[[nodiscard]] inline bool
matches(const registry_model::LeaseSlot &slot,
        const execution::GraphPersist &ticket) noexcept {
  if (!ticket || slot.state != registry_model::LeaseState::Drain ||
      slot.token != ticket.token() || slot.generation != ticket.generation() ||
      slot.coordinate != ticket.coordinate() || ticket.coordinate() == 0u ||
      slot.plan != ticket.plan() || slot.book_domain != ticket.book_domain() ||
      slot.retry_region != ticket.region() ||
      slot.identity != ticket.identity() ||
      slot.transitions.size() != ticket.pages().size() ||
      slot.undo_frames.size() != ticket.pages().size() ||
      slot.undo.size() != ticket.pages().size()) {
    return false;
  }
  const auto pages = ticket.pages();
  for (std::size_t index = 0u; index < pages.size(); ++index) {
    const execution::GraphPersistPage &page = pages[index];
    const CacheTransition &transition = slot.transitions[index];
    if (transition.frame != page.frame || transition.key != page.key ||
        transition.kind != TransitionKind::Writeback ||
        transition.dirty.offset != page.backing_offset ||
        transition.dirty.bytes != page.bytes || page.bytes == 0u) {
      return false;
    }
  }
  return true;
}

[[nodiscard]] inline bool finish(std::vector<registry_model::Frame> &frames,
                                 registry_model::LeaseSlot &slot,
                                 const bool success) noexcept {
  if (!check_undo(frames, slot, UndoMode::Drain)) {
    return false;
  }
  if (success) {
    for (const CacheTransition &transition : slot.transitions) {
      if (transition.kind != TransitionKind::Writeback ||
          transition.frame >= frames.size()) {
        return false;
      }
      const registry_model::Frame &frame = frames[transition.frame];
      if (frame.state != FrameState::Writeback || frame.key != transition.key ||
          frame.dirty != transition.dirty) {
        return false;
      }
    }
    for (const CacheTransition &transition : slot.transitions) {
      frames[transition.frame] = frame_detail::empty(frames[transition.frame]);
    }
  } else {
    rollback(frames, slot);
  }
  ::rund::compute::detail::residency::clear(slot);
  return true;
}

[[nodiscard]] inline bool mint(std::uint64_t &next,
                               std::uint64_t &value) noexcept {
  if (next == 0u || next == std::numeric_limits<std::uint64_t>::max()) {
    return false;
  }
  value = next++;
  return value != 0u;
}

} // namespace rund::compute::detail::residency::graph_persist_detail

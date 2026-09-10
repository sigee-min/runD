#include "../release_check.hpp"

#include "../../execution/graph_persist/internal.hpp"

#include <array>

namespace rund::compute::detail::residency::release_detail {

bool ReleaseCheck::retry(const Authority &authority,
                         const Authority::LeaseSlot &checked) noexcept {
  if (checked.token == 0u || checked.generation == 0u ||
      checked.coordinate == 0u || checked.book_domain == 0u ||
      checked.retry_region.count == 0u ||
      checked.retry_region.tier != FrameTier::Host ||
      checked.retry_region.role != FrameRole::Output ||
      checked.plan == Identity{} || checked.terminal ||
      !checked.identity.valid() ||
      checked.retry_region.first > authority.frames_.size() ||
      checked.retry_region.count >
          authority.frames_.size() - checked.retry_region.first ||
      checked.undo_frames.size() != checked.undo.size() ||
      checked.transitions.size() != checked.undo.size() ||
      checked.transitions.empty()) {
    return false;
  }
  return graph_persist_detail::check_undo(
      authority.frames_, checked, graph_persist_detail::UndoMode::Retry);
}

bool ReleaseCheck::persist(const Authority &authority,
                           const Authority::LeaseSlot &checked) noexcept {
  if (checked.state != Authority::LeaseState::Drain) {
    return true;
  }
  if (checked.token == 0u || checked.generation == 0u ||
      checked.coordinate == 0u || checked.plan == Identity{} ||
      checked.transitions.empty() || checked.retry_region.count == 0u ||
      checked.retry_region.first > authority.frames_.size() ||
      checked.retry_region.count >
          authority.frames_.size() - checked.retry_region.first ||
      checked.transitions.size() != checked.undo_frames.size() ||
      checked.transitions.size() != checked.undo.size()) {
    return false;
  }
  if (checked.book_domain != 0u || checked.identity.valid()) {
    if (checked.book_domain == 0u || !checked.identity.valid() ||
        checked.retry_region.tier != FrameTier::Host ||
        checked.retry_region.role != FrameRole::Output) {
      return false;
    }
  } else if (checked.identity != GraphPersistIdentity{}) {
    return false;
  }
  return graph_persist_detail::check_undo(
      authority.frames_, checked, graph_persist_detail::UndoMode::Drain);
}

bool ReleaseCheck::graph_rows(const Authority &authority) noexcept {
  std::array<std::uint32_t, execution::GraphPersistAggregateCapacity> claims{};
  std::size_t claim_count = 0u;
  for (const Authority::LeaseSlot &row :
       authority.cycle_state_.graph_persists) {
    if (row.state == Authority::LeaseState::Free) {
      if (!idle(row)) {
        return false;
      }
      continue;
    }
    if (row.state == Authority::LeaseState::RetryReady) {
      if (!retry(authority, row)) {
        return false;
      }
    } else if (row.state != Authority::LeaseState::Drain ||
               !persist(authority, row)) {
      return false;
    }
    for (const CacheTransition &transition : row.transitions) {
      if (claim_count >= claims.size() ||
          transition.frame >= authority.frames_.size()) {
        return false;
      }
      for (std::size_t prior = 0u; prior < claim_count; ++prior) {
        if (claims[prior] == transition.frame) {
          return false;
        }
      }
      claims[claim_count++] = transition.frame;
    }
  }
  return true;
}

bool ReleaseCheck::slot(const Authority &authority,
                        const std::span<const FrameRegion> regions,
                        const Authority::LeaseSlot &checked) noexcept {
  if (!active(checked)) {
    return idle(checked);
  }
  if (checked.state == Authority::LeaseState::RetryReady) {
    return retry(authority, checked);
  }
  if (!persist(authority, checked)) {
    return false;
  }
  if (checked.token == 0u ||
      checked.undo_frames.size() != checked.undo.size()) {
    return false;
  }
  bool has_rows = false;
  std::size_t remap_cursor = 0u;
  for (const std::uint32_t frame : checked.undo_frames) {
    if (!row(authority, regions, frame, has_rows)) {
      return false;
    }
  }
  for (const CacheBinding &binding : checked.bindings) {
    if (!row(authority, regions, binding.frame, has_rows)) {
      return false;
    }
  }
  for (const CacheTransition &transition : checked.transitions) {
    if (!row(authority, regions, transition.frame, has_rows)) {
      return false;
    }
  }
  for (const GraphLeasePort &port : checked.ports) {
    if (!region(authority, regions, port.region, has_rows) ||
        port.cache_region_count > port.cache_regions.size() ||
        port.first_binding > checked.bindings.size() ||
        port.binding_count > checked.bindings.size() - port.first_binding ||
        port.first_remap != remap_cursor ||
        port.first_remap > checked.remaps.size() ||
        port.remap_count > checked.remaps.size() - port.first_remap) {
      return false;
    }
    remap_cursor += port.remap_count;
    for (std::size_t index = 0u; index < port.cache_region_count; ++index) {
      if (!region(authority, regions, port.cache_regions[index], has_rows)) {
        return false;
      }
    }
  }
  if (remap_cursor != checked.remaps.size()) {
    return false;
  }
  for (const GraphRelocation &relocation : checked.relocations) {
    if (!row(authority, regions, relocation.source_frame, has_rows) ||
        !row(authority, regions, relocation.target_frame, has_rows)) {
      return false;
    }
  }
  for (const std::uint32_t frame : checked.relocation_frames) {
    if (!row(authority, regions, frame, has_rows)) {
      return false;
    }
  }
  return has_rows;
}

} // namespace rund::compute::detail::residency::release_detail

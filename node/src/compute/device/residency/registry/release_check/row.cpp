#include "../release_check.hpp"

namespace rund::compute::detail::residency::release_detail {

bool ReleaseCheck::row(const Authority &authority,
                       const std::span<const FrameRegion> regions,
                       const std::uint32_t index, bool &has_rows) noexcept {
  if (index >= authority.frames_.size() ||
      !authority.frames_[index].assigned) {
    return false;
  }
  has_rows = true;
  const Authority::Frame &owner = authority.frames_[index];
  for (const FrameRegion region : regions) {
    for (std::size_t requested = region.first;
         requested < static_cast<std::size_t>(region.first) + region.count;
         ++requested) {
      const Authority::Frame &selected = authority.frames_[requested];
      if (requested == index ||
          (selected.extent != 0u && selected.extent == owner.extent)) {
        return false;
      }
    }
  }
  return true;
}

bool ReleaseCheck::direct_binding(
    const Authority &authority,
    const ResidentRecurrenceBinding &binding) noexcept {
  const std::size_t index = binding.region.first;
  if (index >= authority.frames_.size() || binding.registration == 0u ||
      binding.region.count != 1u ||
      binding.region.tier != FrameTier::Device) {
    return false;
  }
  const Authority::Frame &frame = authority.frames_[index];
  return frame.assigned && frame.tier == binding.region.tier &&
         frame.role == binding.region.role &&
         frame.extent == binding.registration &&
         frame.view == binding.registration &&
         frame.key.domain == CacheDomain::Transient &&
         frame.key.backing == binding.view.resource &&
         frame.key.version == binding.view.bytes &&
         frame.key.extent == binding.view.offset_bytes &&
         frame.key.materialization_hi == binding.view.element_bytes &&
         frame.key.materialization_lo == binding.view.stride_bytes &&
         frame.key.page == binding.view.count;
}

bool ReleaseCheck::region(const Authority &authority,
                          const std::span<const FrameRegion> regions,
                          const FrameRegion checked,
                          bool &has_rows) noexcept {
  if (checked.count == 0u || checked.first > authority.frames_.size() ||
      checked.count > authority.frames_.size() - checked.first) {
    return false;
  }
  for (std::size_t index = checked.first;
       index < static_cast<std::size_t>(checked.first) + checked.count;
       ++index) {
    const Authority::Frame &frame = authority.frames_[index];
    if (!frame.assigned || frame.tier != checked.tier ||
        frame.role != checked.role ||
        !row(authority, regions, static_cast<std::uint32_t>(index), has_rows)) {
      return false;
    }
  }
  return true;
}

bool ReleaseCheck::idle(const Authority::LeaseSlot &slot) noexcept {
  return slot.token == 0u && slot.cycle == 0u && slot.generation == 0u &&
         slot.coordinate == 0u && slot.book_domain == 0u &&
         slot.retry_region == FrameRegion{} && !slot.cpu_key &&
         slot.plan == Identity{} && slot.identity == GraphPersistIdentity{} &&
         slot.state == Authority::LeaseState::Free && !slot.terminal &&
         !slot.cpu_bound && slot.undo_frames.empty() && slot.undo.empty() &&
         slot.bindings.empty() && slot.transitions.empty() &&
         slot.ports.empty() && slot.remaps.empty() &&
         slot.relocations.empty() && slot.relocation_frames.empty();
}

bool ReleaseCheck::active(const Authority::LeaseSlot &slot) noexcept {
  return slot.state != Authority::LeaseState::Free;
}

} // namespace rund::compute::detail::residency::release_detail

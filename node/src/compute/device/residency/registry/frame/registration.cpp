#include "../../registry.hpp"

#include "../frame.hpp"
#include "../internal.hpp"
#include "../lease_state.hpp"
#include "../release_check.hpp"

#include <kernel/program/compute/binding/model.hpp>

#include <algorithm>
#include <atomic>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <new>

namespace rund::compute::detail::residency {

namespace {

[[nodiscard]] std::uint64_t next_resident_registration() noexcept {
  static std::atomic<std::uint64_t> next{1u};
  std::uint64_t value = next.fetch_add(1u, std::memory_order_relaxed);
  while (value == 0u) {
    value = next.fetch_add(1u, std::memory_order_relaxed);
  }
  return value;
}

[[nodiscard]] bool
valid_resident_view(const ResidentRecurrenceView view) noexcept {
  constexpr std::uint32_t read_write =
      rund::kernel::kResidentUsageRead | rund::kernel::kResidentUsageWrite;
  if (view.resource == 0u || view.bytes == 0u || view.element_bytes == 0u ||
      view.stride_bytes < view.element_bytes || view.count == 0u ||
      (view.usage != rund::kernel::kResidentUsageRead &&
       view.usage != rund::kernel::kResidentUsageWrite &&
       view.usage != read_write) ||
      view.offset_bytes > view.bytes ||
      view.count - 1u >
          std::numeric_limits<std::uint64_t>::max() / view.stride_bytes) {
    return false;
  }
  const std::uint64_t last = (view.count - 1u) * view.stride_bytes;
  return last <= view.bytes - view.offset_bytes &&
         view.element_bytes <= view.bytes - view.offset_bytes - last;
}

} // namespace

bool Authority::register_frames(const FrameTier tier, const FrameRole role,
                                const std::uint32_t frame_capacity,
                                std::uint32_t &first_frame) noexcept {
  return register_frames(tier, role, frame_capacity, 0u, 0u, first_frame);
}

bool Authority::register_frames(const FrameTier tier, const FrameRole role,
                                const std::uint32_t frame_capacity,
                                const std::uint64_t extent,
                                const std::uint64_t view,
                                std::uint32_t &first_frame) noexcept {
  return register_frames_impl(tier, role, frame_capacity, extent, view,
                              FrameState::Empty, {}, first_frame);
}

bool Authority::register_resident_recurrence_view(
    const FrameRole role, const ResidentRecurrenceView view,
    ResidentRecurrenceBinding &binding) noexcept {
  binding = {};
  constexpr std::uint32_t read_write =
      rund::kernel::kResidentUsageRead | rund::kernel::kResidentUsageWrite;
  if (!valid_resident_view(view) ||
      (role == FrameRole::Input &&
       view.usage != rund::kernel::kResidentUsageRead) ||
      (role == FrameRole::Output &&
       view.usage != rund::kernel::kResidentUsageWrite) ||
      (role == FrameRole::Intermediate && view.usage != read_write)) {
    return false;
  }
  const std::uint64_t registration = next_resident_registration();
  CacheKey key{.backing = view.resource,
               .version = view.bytes,
               .extent = view.offset_bytes,
               .materialization_hi = view.element_bytes,
               .materialization_lo = view.stride_bytes,
               .page = view.count,
               .domain = CacheDomain::Transient};
  std::uint32_t first = 0u;
  if (!register_frames_impl(FrameTier::Device, role, 1u, registration,
                            registration, FrameState::Resident, key, first)) {
    return false;
  }
  binding = ResidentRecurrenceBinding{
      .view = view,
      .region = FrameRegion{.tier = FrameTier::Device,
                            .role = role,
                            .first = first,
                            .count = 1u},
      .registration = registration,
  };
  return true;
}

bool Authority::register_frames_impl(const FrameTier tier, const FrameRole role,
                                     const std::uint32_t frame_capacity,
                                     const std::uint64_t extent,
                                     const std::uint64_t view,
                                     const FrameState state, const CacheKey key,
                                     std::uint32_t &first_frame) noexcept {
  first_frame = 0u;
  if (frame_capacity == 0u || ((extent == 0u) != (view == 0u)) ||
      (state == FrameState::Resident && (extent == 0u || key.backing == 0u))) {
    return false;
  }
  std::lock_guard lock{gate_};
  if (view_commit_state_locked() != registry_model::ViewCommitState::Idle ||
      execution_state_.slot.token != 0u || active(cycle_state_.writeback) ||
      any_active(cycle_state_.graph_persists) ||
      std::any_of(cycle_state_.epochs.begin(), cycle_state_.epochs.end(),
                  active)) {
    return false;
  }
  const std::size_t count = frame_capacity;
  std::size_t first = frames_.size();
  std::size_t run = 0u;
  for (std::size_t index = 0u; index < frames_.size(); ++index) {
    if (!frames_[index].assigned) {
      if (run == 0u) {
        first = index;
      }
      if (++run == count) {
        break;
      }
    } else {
      run = 0u;
      first = frames_.size();
    }
  }
  const bool append = run != count;
  const std::size_t old_size = frames_.size();
  if (append &&
      (count > std::numeric_limits<std::uint32_t>::max() - old_size)) {
    return false;
  }
  if (append && count > std::numeric_limits<std::size_t>::max() - old_size) {
    return false;
  }
  const std::size_t total = append ? old_size + count : frames_.size();
  if (!ensure_view_commit_storage_locked(total)) {
    return false;
  }
  try {
    if (append) {
      first = old_size;
      frames_.resize(old_size + count);
    }
    const std::size_t total = frames_.size();
    for (LeaseSlot &slot : cycle_state_.epochs) {
      slot.undo_frames.reserve(total);
      slot.undo.reserve(total);
      slot.bindings.reserve(total);
      slot.transitions.reserve(total * 4u);
      slot.ports.reserve(TiledGraphPortCapacity);
      slot.remaps.reserve(GraphPageRemapCapacity);
      slot.relocations.reserve(total * 2u);
      slot.relocation_frames.reserve(total);
    }
    cycle_state_.writeback.undo_frames.reserve(total);
    cycle_state_.writeback.undo.reserve(total);
    cycle_state_.writeback.bindings.reserve(total);
    cycle_state_.writeback.transitions.reserve(total);
    cycle_state_.writeback.ports.reserve(TiledGraphPortCapacity);
    cycle_state_.writeback.relocations.reserve(total * 2u);
    cycle_state_.writeback.relocation_frames.reserve(total);
    for (LeaseSlot &slot : cycle_state_.graph_persists) {
      slot.undo_frames.reserve(total);
      slot.undo.reserve(total);
      slot.transitions.reserve(total);
    }
    for (std::size_t index = first; index < first + count; ++index) {
      frames_[index] = Frame{.key = key,
                             .state = state,
                             .tier = tier,
                             .role = role,
                             .extent = extent,
                             .view = view,
                             .assigned = true};
    }
    first_frame = static_cast<std::uint32_t>(first);
    return true;
  } catch (const std::bad_alloc &) {
    if (append) {
      frames_.resize(old_size);
    }
    return false;
  }
}

} // namespace rund::compute::detail::residency

#pragma once

#include "../../registry/direct_recurrence_owner.hpp"

#include <kernel/program/compute/binding/model.hpp>

#include <atomic>

namespace rund::compute::detail::residency::direct_recurrence_detail {

[[nodiscard]] inline std::uint64_t next_owner() noexcept {
  static std::atomic<std::uint64_t> next{1u};
  std::uint64_t value = next.fetch_add(1u, std::memory_order_relaxed);
  while (value == 0u) {
    value = next.fetch_add(1u, std::memory_order_relaxed);
  }
  return value;
}

[[nodiscard]] inline std::uint64_t mint_local(std::uint64_t &next) noexcept {
  const std::uint64_t value = next++;
  if (next == 0u) {
    next = 1u;
  }
  return value;
}

[[nodiscard]] inline bool active(const Authority::LeaseSlot &slot) noexcept {
  return slot.state != Authority::LeaseState::Free;
}

[[nodiscard]] inline bool
exact_view(const Authority::Frame &frame,
           const ResidentRecurrenceBinding &binding) noexcept {
  const ResidentRecurrenceView view = binding.view;
  return binding.registration != 0u && binding.region.count == 1u &&
         binding.region.tier == FrameTier::Device && frame.assigned &&
         frame.tier == binding.region.tier &&
         frame.role == binding.region.role &&
         frame.extent == binding.registration &&
         frame.view == binding.registration &&
         frame.key.domain == CacheDomain::Transient &&
         frame.key.backing == view.resource &&
         frame.key.version == view.bytes &&
         frame.key.extent == view.offset_bytes &&
         frame.key.materialization_hi == view.element_bytes &&
         frame.key.materialization_lo == view.stride_bytes &&
         frame.key.page == view.count;
}

} // namespace rund::compute::detail::residency::direct_recurrence_detail

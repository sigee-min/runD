#include "../../receipts.hpp"

namespace rund::compute::detail::graph_reduce {

bool CpuReceiptBook::occupied(
    const residency::CpuReservationKey key) const noexcept {
  const Slot *const entry = find(key);
  return entry != nullptr && entry->state != SlotState::Free;
}

bool CpuReceiptBook::credential(
    const residency::CpuReservationKey key) const noexcept {
  const Slot *const entry = find(key);
  return entry != nullptr &&
         (entry->state == SlotState::Prepared ||
          entry->state == SlotState::Armed) &&
         entry->token != 0u && entry->generation != 0u;
}

std::uint64_t
CpuReceiptBook::token(const residency::CpuReservationKey key) const noexcept {
  const Slot *const entry = find(key);
  return credential(key) ? entry->token : 0u;
}

std::uint64_t CpuReceiptBook::generation(
    const residency::CpuReservationKey key) const noexcept {
  const Slot *const entry = find(key);
  return credential(key) ? entry->generation : 0u;
}

bool CpuReceiptBook::idle() const noexcept {
  if (live_permits_ != 0u) {
    return false;
  }
  for (const Slot &slot : slots_) {
    if (slot.state != SlotState::Free || slot.permit != nullptr ||
        (slot.handle != nullptr && slot.handle->has_snapshot())) {
      return false;
    }
  }
  return std::none_of(unknowns_.begin(), unknowns_.end(),
                      [](const UnknownCred &slot) { return slot.live; });
}

std::size_t CpuReceiptBook::live_receipts() const noexcept {
  return live_receipts_;
}

std::size_t CpuReceiptBook::live_permits() const noexcept {
  return live_permits_;
}

} // namespace rund::compute::detail::graph_reduce

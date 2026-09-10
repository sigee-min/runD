#include "../../receipts.hpp"

namespace rund::compute::detail::graph_reduce {

void CpuReceiptBook::drop_receipt() noexcept {
  if (live_receipts_ != 0u) {
    --live_receipts_;
  }
}

void CpuReceiptBook::detach_handle(const std::size_t slot,
                                   CpuEpochReceipt *const handle) noexcept {
  if (slot < slots_.size() && slots_[slot].handle == handle) {
    slots_[slot].handle = nullptr;
    drop_receipt();
  }
}

void CpuReceiptBook::move_handle(const std::size_t slot,
                                 CpuEpochReceipt *const from,
                                 CpuEpochReceipt *const to) noexcept {
  if (slot < slots_.size() && slots_[slot].handle == from) {
    slots_[slot].handle = to;
  }
}

void CpuReceiptBook::move_permit(const std::size_t slot,
                                 CpuEpochPermit *const from,
                                 CpuEpochPermit *const to) noexcept {
  if (slot < slots_.size() && slots_[slot].permit == from) {
    slots_[slot].permit = to;
  }
}

void CpuReceiptBook::detach_permit(const std::size_t slot,
                                   CpuEpochPermit *const permit) noexcept {
  if (slot < slots_.size() && slots_[slot].permit == permit) {
    release_permit(slots_[slot]);
  }
}

void CpuReceiptBook::release_permit(Slot &entry) noexcept {
  CpuEpochPermit *const permit = entry.permit;
  if (permit == nullptr) {
    return;
  }
  entry.permit = nullptr;
  if (live_permits_ != 0u) {
    --live_permits_;
  }
  permit->clear_local();
}

void CpuReceiptBook::drop_handles() noexcept {
  for (Slot &slot : slots_) {
    if (slot.handle != nullptr) {
      slot.handle->detach();
    }
  }
}

bool CpuReceiptBook::clear_closed(
    const std::size_t slot, const residency::CpuReservationKey key) noexcept {
  if (slot >= slots_.size()) {
    return false;
  }
  Slot &entry = slots_[slot];
  if (entry.state == SlotState::Free || entry.key != key) {
    return false;
  }
  release_permit(entry);
  reset_cred(entry);
  return true;
}

void CpuReceiptBook::clear(const CpuEpochReceipt &receipt) noexcept {
  Slot *const entry = find(receipt.key_);
  if (entry != nullptr &&
      (entry->state == SlotState::Prepared ||
       entry->state == SlotState::Armed) &&
      entry->handle == &receipt) {
    release_permit(*entry);
    reset_cred(*entry);
  }
}

CpuReceiptBook::Slot *
CpuReceiptBook::find(const residency::CpuReservationKey key) noexcept {
  if (key.domain() != domain_ || key.slot() == 0u ||
      key.slot() - 1u >= slots_.size()) {
    return nullptr;
  }
  Slot &entry = slots_[static_cast<std::size_t>(key.slot() - 1u)];
  return entry.key == key ? &entry : nullptr;
}

const CpuReceiptBook::Slot *
CpuReceiptBook::find(const residency::CpuReservationKey key) const noexcept {
  if (key.domain() != domain_ || key.slot() == 0u ||
      key.slot() - 1u >= slots_.size()) {
    return nullptr;
  }
  const Slot &entry = slots_[static_cast<std::size_t>(key.slot() - 1u)];
  return entry.key == key ? &entry : nullptr;
}

residency::CpuReservationKey
CpuReceiptBook::key(const std::size_t slot) const noexcept {
  return slot < slots_.size() ? slots_[slot].key
                              : residency::CpuReservationKey{};
}

void CpuReceiptBook::reset_cred(Slot &entry) noexcept {
  CpuEpochReceipt *const handle = entry.handle;
  if (handle != nullptr) {
    handle->set_key({});
  }
  residency::Authority *const authority = entry.authority;
  const std::size_t bank = entry.bank;
  const CpuReceiptRole role = entry.role;
  entry = Slot{.authority = authority,
               .handle = handle,
               .bank = bank,
               .role = role,
               .state = SlotState::Free};
}

void CpuReceiptBook::reset(Slot &entry) noexcept {
  CpuEpochReceipt *const handle = entry.handle;
  if (handle != nullptr) {
    handle->detach();
  }
  reset_cred(entry);
}

} // namespace rund::compute::detail::graph_reduce

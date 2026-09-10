#include "../receipts.hpp"

#include "../../../../device/residency/registry.hpp"
#include "../../../../device/residency/registry/cpu_graph_owner.hpp"

#include <limits>

namespace rund::compute::detail::graph_reduce {

CpuReceiptBook::CpuReceiptBook() noexcept
    : domain_(residency::next_cpu_book_domain()) {}

void CpuReceiptBook::UnknownCred::clear() noexcept { *this = {}; }

CpuReceiptBindResult CpuReceiptBook::bind(CpuEpochReceipt &receipt,
                                          residency::Authority &authority,
                                          const std::size_t bank,
                                          const CpuReceiptRole role) noexcept {
  if (!receipt.empty_handle()) {
    return CpuReceiptBindResult::Busy;
  }
  const std::size_t slot = index(bank, role);
  if (slot >= slots_.size()) {
    return CpuReceiptBindResult::Invalid;
  }
  Slot &entry = slots_[slot];
  if (entry.state != SlotState::Free || entry.handle != nullptr ||
      entry.permit != nullptr) {
    return CpuReceiptBindResult::Busy;
  }
  if (entry.authority != nullptr && entry.authority != &authority) {
    return CpuReceiptBindResult::Invalid;
  }
  entry.authority = &authority;
  entry.bank = bank;
  entry.role = role;
  entry.handle = &receipt;
  receipt.attach(*this, slot);
  return CpuReceiptBindResult::Bound;
}

CpuReserveResult CpuReceiptBook::reserve(const std::size_t slot,
                                         residency::Authority &authority,
                                         CpuEpochPermit &permit) noexcept {
  if (permit) {
    return CpuReserveResult::Busy;
  }
  if (slot >= slots_.size() || slots_[slot].state != SlotState::Free) {
    return CpuReserveResult::Busy;
  }
  Slot &entry = slots_[slot];
  if (domain_ == 0u || entry.handle == nullptr || entry.permit != nullptr ||
      entry.authority == nullptr || entry.authority != &authority ||
      authority.cpu_graph().cpu_owner_id() == 0u ||
      authority.cpu_graph().cpu_owner_id() ==
          std::numeric_limits<std::uint64_t>::max()) {
    return CpuReserveResult::Invalid;
  }
  if (next_reservation_ == 0u ||
      next_reservation_ == std::numeric_limits<std::uint64_t>::max()) {
    return CpuReserveResult::Invalid;
  }
  const std::uint64_t nonce = next_reservation_++;
  const residency::CpuReservationKey key =
      authority.cpu_graph().reserve_cpu_key(domain_, slot, nonce);
  if (!key) {
    return CpuReserveResult::Busy;
  }
  entry.key = key;
  entry.state = SlotState::Reserved;
  entry.permit = &permit;
  if (entry.handle != nullptr) {
    entry.handle->set_key(key);
  }
  permit.set(*this, slot, authority, key);
  ++live_permits_;
  return CpuReserveResult::Reserved;
}

bool CpuReceiptBook::reserved_to(
    const std::size_t slot, const residency::CpuReservationKey key,
    const residency::Authority &authority) const noexcept {
  return static_cast<bool>(key) && slot < slots_.size() &&
         slots_[slot].state == SlotState::Reserved && slots_[slot].key == key &&
         slots_[slot].authority == &authority;
}

bool CpuReceiptBook::bound(
    const std::size_t slot, const residency::CpuReservationKey key,
    const residency::Authority &authority) const noexcept {
  return slot < slots_.size() &&
         (slots_[slot].state == SlotState::Prepared ||
          slots_[slot].state == SlotState::Armed) &&
         slots_[slot].key == key && slots_[slot].authority == &authority;
}

bool CpuReceiptBook::owns(const std::size_t slot,
                          const CpuEpochReceipt *const handle) const noexcept {
  return slot < slots_.size() && slots_[slot].handle == handle;
}

bool CpuReceiptBook::cancel(const std::size_t slot,
                            const residency::CpuReservationKey key,
                            residency::Authority &authority) noexcept {
  if (slot >= slots_.size()) {
    return false;
  }
  Slot &entry = slots_[slot];
  if (entry.state != SlotState::Reserved || entry.key != key ||
      entry.authority != &authority ||
      !authority.cpu_graph().cancel_cpu_reservation(key)) {
    return false;
  }
  release_permit(entry);
  reset_cred(entry);
  return true;
}

void CpuReceiptBook::prepare(const std::size_t slot,
                             const residency::CpuReservationKey key,
                             const std::uint64_t token,
                             const std::uint64_t generation) noexcept {
  if (slot >= slots_.size()) {
    return;
  }
  Slot &entry = slots_[slot];
  if (entry.state != SlotState::Reserved || entry.key != key || token == 0u ||
      generation == 0u) {
    return;
  }
  entry.token = token;
  entry.generation = generation;
  entry.state = SlotState::Prepared;
  if (entry.handle != nullptr) {
    entry.handle->set_credential(token, generation);
  }
}

bool CpuReceiptBook::finalize(const std::size_t slot,
                              const residency::CpuReservationKey key,
                              const std::uint64_t token,
                              const std::uint64_t generation) noexcept {
  if (slot >= slots_.size()) {
    return false;
  }
  Slot &entry = slots_[slot];
  if (entry.state != SlotState::Prepared || entry.key != key ||
      entry.token != token || entry.generation != generation) {
    return false;
  }
  entry.state = SlotState::Armed;
  return true;
}

} // namespace rund::compute::detail::graph_reduce

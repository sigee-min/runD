#include "internal.hpp"

namespace rund::compute::detail::residency {

registry_model::CpuQuarantineOwner::CpuQuarantineOwner(
    Authority &authority) noexcept
    : authority_(authority) {}

registry_model::CpuQuarantineOwner::BookSnapshot
registry_model::CpuQuarantineOwner::snap_q(
    const graph_reduce::CpuReceiptBook &book) noexcept {
  BookSnapshot snap{};
  snap.domain = book.domain_;
  snap.live_receipts = book.live_receipts_;
  snap.live_permits = book.live_permits_;
  snap.valid = book.valid();
  for (std::size_t index = 0u; index < book.slots_.size(); ++index) {
    const graph_reduce::CpuReceiptBook::Slot &item = book.slots_[index];
    BookSnapshot::Epoch &row = snap.epochs[index];
    row.state = item.state;
    row.authority = item.authority;
    row.key = item.key;
    row.token = item.token;
    row.generation = item.generation;
    row.permit = item.permit;
    row.handle = item.handle;
    row.bank = item.bank;
    row.role = static_cast<std::uint8_t>(item.role);
  }
  for (std::size_t index = 0u; index < book.unknowns_.size(); ++index) {
    const graph_reduce::CpuReceiptBook::UnknownCred &item =
        book.unknowns_[index];
    BookSnapshot::Unknown &row = snap.unknowns[index];
    row.authority = item.authority;
    row.plan = item.plan;
    row.identity = item.identity;
    row.region = item.region;
    row.page_count = item.page_count;
    row.domain = item.domain;
    row.token = item.token;
    row.generation = item.generation;
    row.coordinate = item.coordinate;
    row.terminal = item.terminal;
    row.quarantined = item.quarantined;
    row.live = item.live;
  }
  return snap;
}

} // namespace rund::compute::detail::residency

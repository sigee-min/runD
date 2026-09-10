#include "local.hpp"

#include "../frame.hpp"

#include <limits>
#include <mutex>

namespace rund::compute::detail::residency::transaction_detail {

void consume(VirtualTransactionLease &lease) noexcept {
  if (lease.gate.owns_lock()) {
    lease.gate.unlock();
  }
  lease.gate = std::unique_lock<std::mutex>{};
  lease.frames.fill(std::numeric_limits<std::uint32_t>::max());
  lease.row_count = 0u;
  lease.active = false;
  lease.authority = nullptr;
}

} // namespace rund::compute::detail::residency::transaction_detail

namespace rund::compute::detail::residency {

void VirtualTransactionOwner::commit_virtual_transaction_lease(
    VirtualTransactionLease &&lease,
    const std::uint64_t committed_version) const noexcept {
  if (lease.authority != &authority_ || !lease.active ||
      !lease.gate.owns_lock()) {
    transaction_detail::consume(lease);
    return;
  }
  if (authority_.view_commit_quarantined_locked()) {
    transaction_detail::consume(lease);
    return;
  }
  for (std::size_t index = 0u; index < lease.row_count; ++index) {
    Authority::Frame &frame = authority_.frames_[lease.frames[index]];
    frame.key.version = committed_version;
    frame.dirty = {};
    frame.state = FrameState::Resident;
    frame.transaction_generation = 0u;
  }
  transaction_detail::consume(lease);
}

void VirtualTransactionOwner::abort_virtual_transaction_lease(
    VirtualTransactionLease &&lease, const bool unknown) const noexcept {
  (void)unknown;
  if (lease.authority != &authority_ || !lease.active ||
      !lease.gate.owns_lock()) {
    transaction_detail::consume(lease);
    return;
  }
  if (authority_.view_commit_quarantined_locked()) {
    transaction_detail::consume(lease);
    return;
  }
  for (std::size_t index = 0u; index < lease.row_count; ++index) {
    const std::uint32_t frame = lease.frames[index];
    authority_.frames_[frame] = frame_detail::empty(authority_.frames_[frame]);
  }
  transaction_detail::consume(lease);
}

} // namespace rund::compute::detail::residency

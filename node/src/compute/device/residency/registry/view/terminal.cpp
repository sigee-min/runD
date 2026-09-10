#include "../freeze/internal.hpp"
#include "../view_owner.hpp"

#include <cstddef>
#include <memory>
#include <mutex>

namespace rund::compute::detail::residency {

bool ViewOwner::validate_view_commit(
    const ViewCommitReceipt &receipt) const noexcept {
  if (!authority_.view_commit_active_locked(receipt) ||
      receipt.authority_ != &authority_ ||
      receipt.authority_id_ != authority_.credentials_.owner_id ||
      receipt.rows_ == nullptr || receipt.row_count == 0u ||
      receipt.row_count > authority_.frames_.size()) {
    return false;
  }
  for (std::size_t index = 0u; index < receipt.row_count; ++index) {
    const ViewCommitReceipt::Row &row = receipt.rows_[index];
    if (row.frame >= authority_.frames_.size() || row.stamp != receipt.stamp_) {
      return false;
    }
    for (std::size_t prior = 0u; prior < index; ++prior) {
      if (receipt.rows_[prior].frame == row.frame) {
        return false;
      }
    }
    const Frame &frame = authority_.frames_[row.frame];
    if (frame.view_commit_stamp != row.stamp || frame.key != row.post_key ||
        frame.dirty != row.post_dirty || frame.state != row.post_state ||
        frame.tier != row.post_tier || frame.role != row.post_role ||
        frame.extent != row.post_extent || frame.view != row.post_view ||
        frame.next_use != row.post_next_use ||
        frame.retain_until != row.post_retain_until ||
        frame.alias_claims != row.post_alias_claims ||
        frame.transaction_generation != row.post_transaction_generation ||
        frame.direct_registration != row.post_direct_registration ||
        frame.assigned != row.assigned) {
      return false;
    }
    if (row.evicted &&
        (row.post_state != FrameState::Empty || row.post_key != CacheKey{} ||
         !row.post_dirty.empty() || row.post_alias_claims != 0u ||
         row.post_transaction_generation != 0u ||
         row.post_direct_registration != nullptr)) {
      return false;
    }
  }
  return true;
}

bool ViewOwner::close_view_commit(
    std::unique_ptr<ViewCommitReceipt> &receipt) noexcept {
  std::lock_guard lock{authority_.gate_};
  if (receipt == nullptr || authority_.view_commit_quarantined_locked()) {
    return false;
  }
  if (!validate_view_commit(*receipt)) {
    static_cast<void>(move_view_commit_to_quarantine_locked(receipt));
    return false;
  }
  return recycle_view_commit_locked(receipt);
}

bool ViewOwner::abort_view_commit(
    std::unique_ptr<ViewCommitReceipt> &receipt) noexcept {
  std::lock_guard lock{authority_.gate_};
  if (receipt == nullptr || authority_.view_commit_quarantined_locked()) {
    return false;
  }
  if (!validate_view_commit(*receipt)) {
    static_cast<void>(move_view_commit_to_quarantine_locked(receipt));
    return false;
  }
  if (!view_commit_recycle_ready_locked(*receipt)) {
    return false;
  }
  for (std::size_t index = 0u; index < receipt->row_count; ++index) {
    const ViewCommitReceipt::Row &row = receipt->rows_[index];
    if (!row.evicted) {
      Frame &frame = authority_.frames_[row.frame];
      frame.next_use = row.prior_next_use;
      frame.retain_until = row.prior_retain_until;
    }
  }
  return recycle_view_commit_locked(receipt);
}

bool ViewOwner::move_view_commit_to_quarantine_locked(
    std::unique_ptr<ViewCommitReceipt> &receipt) noexcept {
  if (receipt == nullptr || authority_.view_commit_quarantined_locked() ||
      !authority_.view_commit_active_locked(*receipt) ||
      receipt->authority_ != &authority_ ||
      receipt->authority_id_ != authority_.credentials_.owner_id) {
    return false;
  }
  receipt->quarantined_ = true;
  authority_.view_state_.view_quarantine = std::move(receipt);
  return true;
}

bool ViewOwner::quarantine_view_commit(
    std::unique_ptr<ViewCommitReceipt> &receipt) noexcept {
  if (receipt == nullptr) {
    return false;
  }
  std::lock_guard lock{authority_.gate_};
  if (!move_view_commit_to_quarantine_locked(receipt)) {
    return false;
  }
  return receipt == nullptr;
}

} // namespace rund::compute::detail::residency

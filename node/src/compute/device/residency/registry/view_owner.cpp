#include "view_owner.hpp"

#include <utility>

namespace rund::compute::detail::residency {

ViewOwner::ViewOwner(Authority &authority) noexcept : authority_(authority) {}

ViewOwner Authority::views() noexcept { return ViewOwner{*this}; }

bool ViewOwner::view_commit_recycle_ready_locked(
    const ViewCommitReceipt &receipt) const noexcept {
  return authority_.view_state_.view_idle == nullptr &&
         authority_.view_commit_active_locked(receipt) &&
         receipt.authority_ == &authority_ &&
         receipt.authority_id_ == authority_.credentials_.owner_id &&
         receipt.rows_ != nullptr && receipt.row_count != 0u &&
         receipt.row_count <= receipt.row_capacity_;
}

bool ViewOwner::recycle_view_commit_locked(
    std::unique_ptr<ViewCommitReceipt> &receipt) noexcept {
  if (receipt == nullptr || !view_commit_recycle_ready_locked(*receipt)) {
    return false;
  }
  for (std::size_t index = 0u; index < receipt->row_count; ++index) {
    authority_.frames_[receipt->rows_[index].frame].view_commit_stamp = 0u;
  }
  authority_.view_state_.active_view_commit_stamp = 0u;
  receipt->recycle();
  authority_.view_state_.view_idle = std::move(receipt);
  return true;
}

} // namespace rund::compute::detail::residency

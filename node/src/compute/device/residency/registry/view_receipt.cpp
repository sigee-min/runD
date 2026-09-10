#include "view_receipt.hpp"

#include <exception>
#include <utility>

namespace rund::compute::detail::residency::registry_model {

ViewCommitReceipt::ViewCommitReceipt(Authority *const authority,
                                     const std::uint64_t authority_id,
                                     const std::uint64_t stamp) noexcept
    : authority_(authority), authority_id_(authority_id), stamp_(stamp),
      active_(false) {}

ViewCommitReceipt::ViewCommitReceipt(ViewCommitReceipt &&other) noexcept
    : authority_(other.authority_), authority_id_(other.authority_id_),
      stamp_(other.stamp_), rows_(std::move(other.rows_)),
      row_capacity_(other.row_capacity_), row_count(other.row_count),
      active_(other.active_), quarantined_(other.quarantined_) {
  other.drop();
}

ViewCommitReceipt::~ViewCommitReceipt() {
  if (active_ && !quarantined_) {
    std::terminate();
  }
}

void ViewCommitReceipt::recycle() noexcept {
  if (rows_ != nullptr) {
    for (std::size_t index = 0u; index < row_capacity_; ++index) {
      rows_[index] = Row{};
    }
  }
  stamp_ = 0u;
  row_count = 0u;
  active_ = false;
  quarantined_ = false;
}

void ViewCommitReceipt::drop() noexcept {
  authority_ = nullptr;
  authority_id_ = 0u;
  stamp_ = 0u;
  rows_.reset();
  row_capacity_ = 0u;
  row_count = 0u;
  active_ = false;
  quarantined_ = false;
}

} // namespace rund::compute::detail::residency::registry_model

#include "../receipts.hpp"

namespace rund::compute::detail::graph_reduce {

CpuEpochPermit::~CpuEpochPermit() noexcept {
  cancel();
  if (book_ != nullptr) {
    detach();
  }
}

void CpuEpochPermit::set(CpuReceiptBook &book, const std::size_t slot,
                         residency::Authority &authority,
                         const residency::CpuReservationKey key) noexcept {
  book_ = &book;
  authority_ = &authority;
  slot_ = slot;
  key_ = key;
}

void CpuEpochPermit::clear_local() noexcept {
  book_ = nullptr;
  authority_ = nullptr;
  slot_ = 0u;
  key_ = {};
}

CpuEpochPermit::CpuEpochPermit(CpuEpochPermit &&other) noexcept
    : book_(other.book_), authority_(other.authority_), slot_(other.slot_),
      key_(other.key_) {
  if (book_ != nullptr) {
    book_->move_permit(slot_, &other, this);
  }
  other.book_ = nullptr;
  other.authority_ = nullptr;
  other.slot_ = 0u;
  other.key_ = {};
}

CpuEpochPermit &
CpuEpochPermit::operator=(CpuEpochPermit &&other) noexcept {
  if (this != &other) {
    detach();
    book_ = other.book_;
    authority_ = other.authority_;
    slot_ = other.slot_;
    key_ = other.key_;
    if (book_ != nullptr) {
      book_->move_permit(slot_, &other, this);
    }
    other.book_ = nullptr;
    other.authority_ = nullptr;
    other.slot_ = 0u;
    other.key_ = {};
  }
  return *this;
}

bool CpuEpochPermit::bound_to(
    const residency::Authority &authority) const noexcept {
  return book_ != nullptr && authority_ == &authority &&
         book_->reserved_to(slot_, key_, authority);
}

void CpuEpochPermit::cancel() noexcept {
  if (book_ == nullptr || authority_ == nullptr) {
    return;
  }
  if (book_->cancel(slot_, key_, *authority_)) {
    detach();
  }
}

void CpuEpochPermit::prepare(const std::uint64_t token,
                             const std::uint64_t generation) noexcept {
  if (book_ != nullptr) {
    book_->prepare(slot_, key_, token, generation);
  }
}

void CpuEpochPermit::detach() noexcept {
  if (book_ != nullptr) {
    book_->detach_permit(slot_, this);
  }
  book_ = nullptr;
  authority_ = nullptr;
  slot_ = 0u;
  key_ = {};
}

bool CpuEpochPermit::finalize(const std::uint64_t token,
                              const std::uint64_t generation) noexcept {
  return book_ != nullptr && book_->finalize(slot_, key_, token, generation);
}

void CpuEpochPermit::release_closed() noexcept {
  if (book_ == nullptr) {
    return;
  }
  if (book_->clear_closed(slot_, key_)) {
    detach();
  }
}

} // namespace rund::compute::detail::graph_reduce

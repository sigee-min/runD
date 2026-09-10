#include "../receipts.hpp"

namespace rund::compute::detail::graph_reduce {

CpuEpochReceipt::~CpuEpochReceipt() noexcept { detach(); }

CpuEpochReceipt::operator bool() const noexcept {
  return book_ != nullptr && book_->owns(slot_, this) &&
         book_->credential(key_);
}

bool CpuEpochReceipt::occupied() const noexcept {
  return book_ != nullptr && book_->owns(slot_, this) &&
         book_->occupied(key_);
}

std::uint64_t CpuEpochReceipt::token() const noexcept {
  return book_ != nullptr && book_->owns(slot_, this) ? token_ : 0u;
}

std::uint64_t CpuEpochReceipt::generation() const noexcept {
  return book_ != nullptr && book_->owns(slot_, this) ? generation_ : 0u;
}

residency::CpuReservationKey CpuEpochReceipt::key() const noexcept {
  return book_ != nullptr && book_->owns(slot_, this)
             ? key_
             : residency::CpuReservationKey{};
}

bool CpuEpochReceipt::bound_to(
    const residency::Authority &authority) const noexcept {
  return book_ != nullptr && book_->owns(slot_, this) &&
         book_->bound(slot_, key_, authority);
}

CpuEpochReceipt::CpuEpochReceipt(CpuEpochReceipt &&other) noexcept
    : book_(other.book_), slot_(other.slot_), key_(other.key_),
      token_(other.token_), generation_(other.generation_) {
  if (book_ != nullptr) {
    book_->move_handle(slot_, &other, this);
  }
  other.book_ = nullptr;
  other.slot_ = 0u;
  other.key_ = {};
  other.token_ = 0u;
  other.generation_ = 0u;
}

CpuEpochReceipt &
CpuEpochReceipt::operator=(CpuEpochReceipt &&other) noexcept {
  if (this != &other) {
    detach();
    book_ = other.book_;
    slot_ = other.slot_;
    key_ = other.key_;
    token_ = other.token_;
    generation_ = other.generation_;
    if (book_ != nullptr) {
      book_->move_handle(slot_, &other, this);
    }
    other.book_ = nullptr;
    other.slot_ = 0u;
    other.key_ = {};
    other.token_ = 0u;
    other.generation_ = 0u;
  }
  return *this;
}

CpuReserveResult CpuEpochReceipt::reserve(
    residency::Authority &authority, CpuEpochPermit &permit) noexcept {
  return book_ == nullptr ? CpuReserveResult::Invalid
                          : book_->reserve(slot_, authority, permit);
}

bool CpuEpochReceipt::empty_handle() const noexcept { return book_ == nullptr; }

bool CpuEpochReceipt::has_snapshot() const noexcept {
  return book_ != nullptr && static_cast<bool>(key_);
}

void CpuEpochReceipt::set_key(
    const residency::CpuReservationKey key) noexcept {
  key_ = key;
  token_ = 0u;
  generation_ = 0u;
}

void CpuEpochReceipt::set_credential(const std::uint64_t token,
                                     const std::uint64_t generation) noexcept {
  token_ = token;
  generation_ = generation;
}

void CpuEpochReceipt::clear() noexcept {
  if (book_ != nullptr) {
    book_->clear(*this);
  }
}

void CpuEpochReceipt::attach(CpuReceiptBook &book,
                             const std::size_t slot) noexcept {
  book_ = &book;
  slot_ = slot;
  key_ = {};
  token_ = 0u;
  generation_ = 0u;
  ++book.live_receipts_;
}

void CpuEpochReceipt::detach() noexcept {
  if (book_ != nullptr) {
    book_->detach_handle(slot_, this);
  }
  book_ = nullptr;
  slot_ = 0u;
  key_ = {};
  token_ = 0u;
  generation_ = 0u;
}

} // namespace rund::compute::detail::graph_reduce

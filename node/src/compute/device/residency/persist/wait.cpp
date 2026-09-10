#include "../persist.hpp"

namespace rund::compute::detail::residency {

PersistReceipt Persister::wait() noexcept {
  std::unique_lock lock{gate_};
  if (state_ == State::Pending) {
    ready_.wait(lock, [this] { return state_ != State::Pending; });
  }
  if (state_ != State::Ready) {
    return {};
  }
  PersistReceipt receipt{
      .status = status_,
      .pages = std::span<const PersistedPage>{pages_.data(), count_},
      .token = token_,
      .io_ns = io_ns_,
      .may_write = may_write_,
  };
  backing_ = nullptr;
  count_ = 0u;
  token_ = 0u;
  may_write_ = false;
  state_ = State::Idle;
  return receipt;
}

} // namespace rund::compute::detail::residency

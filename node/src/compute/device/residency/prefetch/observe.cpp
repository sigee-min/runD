#include "local.hpp"

#include <rund/counter.hpp>

namespace rund::compute::detail::residency {

bool Prefetcher::ready() const noexcept {
  std::lock_guard lock{gate_};
  return state_ == State::Ready;
}

PrefetchReceipt Prefetcher::wait() noexcept {
  const std::uint64_t started = prefetch_detail::now_ns();
  std::unique_lock lock{gate_};
  if (state_ == State::Pending) {
    ready_.wait(lock, [this] { return state_ != State::Pending; });
  }
  if (state_ != State::Ready) {
    return PrefetchReceipt{};
  }
  PrefetchReceipt receipt{
      .status = status_,
      .pages = std::span<const PrefetchedPage>{pages_.data(), count_},
      .token = token_,
      .io_ns = io_ns_,
      .wait_ns = prefetch_detail::now_ns() - started,
      .speculative = speculative_,
      .coherent_input = coherent_input_,
      .coherent_deferred = coherent_deferred_,
      .aliases = std::span<AliasLease>{aliases_.data(), alias_count_},
  };
  backing_ = nullptr;
  token_ = 0u;
  speculative_ = false;
  coherent_input_ = false;
  coherent_deferred_ = false;
  state_ = State::Idle;
  return receipt;
}

bool Prefetcher::quiescent() const noexcept {
  std::lock_guard lock{gate_};
  return (state_ == State::Empty || state_ == State::Idle) &&
         backing_ == nullptr && token_ == 0u && alias_count_ == 0u;
}

std::uint64_t Prefetcher::retained_host_bytes() const noexcept {
  std::lock_guard lock{gate_};
  const auto capacity_bytes = [](const std::size_t capacity,
                                 const std::size_t width) noexcept {
    return ::rund::detail::counter::SaturatingMultiply(
        static_cast<std::uint64_t>(capacity),
        static_cast<std::uint64_t>(width));
  };
  std::uint64_t bytes =
      capacity_bytes(requests_.capacity(), sizeof(PrefetchRequest));
  bytes = ::rund::detail::counter::SaturatingAdd(
      bytes, capacity_bytes(pages_.capacity(), sizeof(PrefetchedPage)));
  return ::rund::detail::counter::SaturatingAdd(
      bytes, capacity_bytes(aliases_.capacity(), sizeof(AliasLease)));
}

} // namespace rund::compute::detail::residency

#include "prefetch/local.hpp"

#include <limits>
#include <new>
#include <system_error>

namespace rund::compute::detail::residency {

Prefetcher::~Prefetcher() {
  {
    std::lock_guard lock{gate_};
    if (state_ != State::Empty) {
      if (state_ != State::Idle || backing_ != nullptr || token_ != 0u ||
          alias_count_ != 0u) {
        std::terminate();
      }
      state_ = State::Stop;
    }
  }
  pending_.notify_one();
  if (worker_.joinable()) {
    worker_.join();
  }
}

bool Prefetcher::configure(const std::uint64_t page_bytes,
                           const std::uint32_t capacity,
                           PrefetchCompletion *const completion) noexcept {
  if (page_bytes == 0u || capacity == 0u ||
      page_bytes > std::numeric_limits<std::size_t>::max()) {
    return false;
  }
  std::lock_guard lock{gate_};
  if (state_ != State::Empty) {
    return page_bytes_ == page_bytes && requests_.size() == capacity &&
           completion_ == completion;
  }
  try {
    requests_.resize(capacity);
    pages_.resize(capacity);
    aliases_.resize(capacity);
    page_bytes_ = page_bytes;
    completion_ = completion;
    state_ = State::Idle;
    worker_ = std::thread{[this] { work(); }};
    return true;
  } catch (const std::bad_alloc &) {
    requests_.clear();
    pages_.clear();
    aliases_.clear();
    state_ = State::Empty;
    return false;
  } catch (const std::system_error &) {
    requests_.clear();
    pages_.clear();
    aliases_.clear();
    state_ = State::Empty;
    return false;
  }
}

} // namespace rund::compute::detail::residency

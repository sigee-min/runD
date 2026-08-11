#include "residency_prefetch.hpp"

#include <rund/compute/virtual.hpp>

#include <chrono>
#include <cstring>
#include <limits>
#include <new>
#include <system_error>

namespace rund::compute::detail::residency {
namespace {

[[nodiscard]] std::uint64_t now_ns() noexcept {
  return static_cast<std::uint64_t>(
      std::chrono::duration_cast<std::chrono::nanoseconds>(
          std::chrono::steady_clock::now().time_since_epoch())
          .count());
}

} // namespace

Prefetcher::~Prefetcher() {
  {
    std::lock_guard lock{gate_};
    if (state_ != State::Empty) {
      state_ = State::Stop;
    }
  }
  pending_.notify_one();
  if (worker_.joinable()) {
    worker_.join();
  }
}

bool Prefetcher::configure(const std::uint64_t page_bytes,
                           const std::uint32_t capacity) noexcept {
  if (page_bytes == 0u || capacity == 0u ||
      page_bytes > std::numeric_limits<std::size_t>::max() ||
      capacity > std::numeric_limits<std::size_t>::max() / page_bytes) {
    return false;
  }
  std::lock_guard lock{gate_};
  if (state_ != State::Empty) {
    return page_bytes_ == page_bytes && requests_.size() == capacity;
  }
  try {
    requests_.resize(capacity);
    pages_.resize(capacity);
    ranges_.resize(capacity);
    storage_.resize(static_cast<std::size_t>(page_bytes) * capacity);
    page_bytes_ = page_bytes;
    state_ = State::Idle;
    worker_ = std::thread{[this] { work(); }};
    return true;
  } catch (const std::bad_alloc &) {
    requests_.clear();
    pages_.clear();
    ranges_.clear();
    storage_.clear();
    state_ = State::Empty;
    return false;
  } catch (const std::system_error &) {
    requests_.clear();
    pages_.clear();
    ranges_.clear();
    storage_.clear();
    state_ = State::Empty;
    return false;
  }
}

bool Prefetcher::submit(
    VirtualBacking &backing,
    const std::span<const PrefetchRequest> requests) noexcept {
  std::lock_guard lock{gate_};
  if (state_ != State::Idle || requests.empty() ||
      requests.size() > requests_.size()) {
    return false;
  }
  for (std::size_t index = 0u; index < requests.size(); ++index) {
    if (requests[index].key.backing == 0u || requests[index].bytes == 0u ||
        requests[index].target_offset > page_bytes_ ||
        requests[index].bytes > page_bytes_ - requests[index].target_offset) {
      return false;
    }
    requests_[index] = requests[index];
    pages_[index] = PrefetchedPage{
        .key = requests[index].key,
        .bytes = requests[index].bytes,
        .storage_offset = static_cast<std::size_t>(page_bytes_) * index,
        .target_offset = requests[index].target_offset,
    };
  }
  backing_ = &backing;
  count_ = requests.size();
  status_ = Status::success();
  io_ns_ = 0u;
  state_ = State::Pending;
  pending_.notify_one();
  return true;
}

PrefetchReceipt Prefetcher::wait() noexcept {
  const std::uint64_t started = now_ns();
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
      .storage = storage_.data(),
      .io_ns = io_ns_,
      .wait_ns = now_ns() - started,
  };
  backing_ = nullptr;
  state_ = State::Idle;
  return receipt;
}

std::uint64_t Prefetcher::storage_bytes() const noexcept {
  std::lock_guard lock{gate_};
  return storage_.size();
}

void Prefetcher::work() noexcept {
  for (;;) {
    std::unique_lock lock{gate_};
    pending_.wait(lock, [this] {
      return state_ == State::Pending || state_ == State::Stop;
    });
    if (state_ == State::Stop) {
      return;
    }
    VirtualBacking *const backing = backing_;
    const std::size_t count = count_;
    for (std::size_t index = 0u; index < count; ++index) {
      std::memset(storage_.data() + pages_[index].storage_offset, 0,
                  static_cast<std::size_t>(page_bytes_));
      ranges_[index] = VirtualRead{
          .offset = requests_[index].offset,
          .bytes = std::span<std::byte>{storage_.data() +
                                            pages_[index].storage_offset +
                                            pages_[index].target_offset,
                                        pages_[index].bytes},
      };
    }
    lock.unlock();
    const std::uint64_t started = now_ns();
    const Status status =
        backing == nullptr ? Status::fail(Reason::PipelineInvalid)
                           : backing->read_batch(std::span<const VirtualRead>{
                                 ranges_.data(), count});
    const std::uint64_t elapsed = now_ns() - started;
    lock.lock();
    status_ = status;
    io_ns_ = elapsed;
    state_ = State::Ready;
    ready_.notify_one();
  }
}

} // namespace rund::compute::detail::residency

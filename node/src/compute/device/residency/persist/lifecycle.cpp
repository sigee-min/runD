#include "../persist.hpp"

#include <rund/counter.hpp>

#include <limits>
#include <new>
#include <system_error>

namespace rund::compute::detail::residency {

Persister::~Persister() {
  {
    std::lock_guard lock{gate_};
    if (state_ != State::Empty) {
      if (state_ != State::Idle || backing_ != nullptr || token_ != 0u) {
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

bool Persister::configure(const std::uint64_t frame_bytes,
                          const std::uint32_t capacity) noexcept {
  if (frame_bytes == 0u || capacity == 0u ||
      frame_bytes > std::numeric_limits<std::size_t>::max()) {
    return false;
  }
  std::lock_guard lock{gate_};
  if (state_ != State::Empty) {
    return frame_bytes_ == frame_bytes && requests_.size() == capacity;
  }
  try {
    requests_.resize(capacity);
    pages_.resize(capacity);
    frame_bytes_ = frame_bytes;
    state_ = State::Idle;
    worker_ = std::thread{[this] { work(); }};
    return true;
  } catch (const std::bad_alloc &) {
  } catch (const std::system_error &) {
  }
  requests_.clear();
  pages_.clear();
  state_ = State::Empty;
  return false;
}

bool Persister::quiescent() const noexcept {
  std::lock_guard lock{gate_};
  return (state_ == State::Empty || state_ == State::Idle) &&
         backing_ == nullptr && token_ == 0u;
}

std::uint64_t Persister::retained_host_bytes() const noexcept {
  std::lock_guard lock{gate_};
  const auto bytes = [](const std::size_t capacity,
                        const std::size_t width) noexcept {
    return ::rund::detail::counter::SaturatingMultiply(
        static_cast<std::uint64_t>(capacity),
        static_cast<std::uint64_t>(width));
  };
  return ::rund::detail::counter::SaturatingAdd(
      bytes(requests_.capacity(), sizeof(PersistRequest)),
      bytes(pages_.capacity(), sizeof(PersistedPage)));
}

} // namespace rund::compute::detail::residency

#pragma once

#include <rund/counter.hpp>

#include <chrono>
#include <cstdint>

namespace rund::node::accel::detail {

[[nodiscard]] inline std::uint64_t MonotonicNanoseconds() noexcept {
  return static_cast<std::uint64_t>(
      std::chrono::duration_cast<std::chrono::nanoseconds>(
          std::chrono::steady_clock::now().time_since_epoch())
          .count());
}

class ReadbackTimer final {
public:
  explicit ReadbackTimer(std::uint64_t &counter) noexcept
      : counter_(&counter), begin_(MonotonicNanoseconds()) {}

  ReadbackTimer(const ReadbackTimer &) = delete;
  ReadbackTimer &operator=(const ReadbackTimer &) = delete;

  ~ReadbackTimer() {
    ::rund::detail::counter::Accumulate(*counter_,
                                        MonotonicNanoseconds() - begin_);
  }

private:
  std::uint64_t *counter_;
  std::uint64_t begin_;
};

} // namespace rund::node::accel::detail

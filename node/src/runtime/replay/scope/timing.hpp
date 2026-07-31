#pragma once

#include <rund/telemetry/event.hpp>

#include <chrono>
#include <cstdint>

namespace rund::replay::detail::scope {

// One clock spans the complete public scope operation. The runtime marks the
// callback boundaries; the facade stops the final publication interval.
class Timing final {
public:
  explicit Timing(const bool enabled) noexcept : enabled_(enabled) {
    prepare_begin_ = read();
  }

  Timing(const Timing &) = delete;
  Timing &operator=(const Timing &) = delete;

  void work() noexcept {
    if (phase_ != Phase::Prepare) {
      return;
    }
    work_begin_ = read();
    phase_ = Phase::Work;
  }

  void finish() noexcept {
    if (phase_ != Phase::Work) {
      return;
    }
    finish_begin_ = read();
    phase_ = Phase::Finish;
  }

  [[nodiscard]] ::rund::telemetry::Detail stop() noexcept {
    if (phase_ != Phase::Stopped) {
      end_ = read();
      phase_ = Phase::Stopped;
    }
    return detail();
  }

  [[nodiscard]] ::rund::telemetry::Detail detail() const noexcept {
    if (!enabled_) {
      return {};
    }
    switch (phase_) {
    case Phase::Prepare:
      return {};
    case Phase::Work:
      return {.prepare_ns = elapsed(prepare_begin_, work_begin_)};
    case Phase::Finish:
      return {
          .prepare_ns = elapsed(prepare_begin_, work_begin_),
          .work_ns = elapsed(work_begin_, finish_begin_),
      };
    case Phase::Stopped:
      if (finish_begin_ != Time{}) {
        return {
            .prepare_ns = elapsed(prepare_begin_, work_begin_),
            .work_ns = elapsed(work_begin_, finish_begin_),
            .finish_ns = elapsed(finish_begin_, end_),
        };
      }
      if (work_begin_ != Time{}) {
        return {
            .prepare_ns = elapsed(prepare_begin_, work_begin_),
            .work_ns = elapsed(work_begin_, end_),
        };
      }
      return {.prepare_ns = elapsed(prepare_begin_, end_)};
    }
    return {};
  }

  [[nodiscard]] std::uint8_t reads() const noexcept { return reads_; }

private:
  using Clock = std::chrono::steady_clock;
  using Time = Clock::time_point;

  enum class Phase : std::uint8_t {
    Prepare,
    Work,
    Finish,
    Stopped,
  };

  [[nodiscard]] Time read() noexcept {
    if (!enabled_) {
      return {};
    }
    ++reads_;
    return Clock::now();
  }

  [[nodiscard]] static std::uint64_t elapsed(const Time begin,
                                             const Time end) noexcept {
    if (end <= begin) {
      return 0u;
    }
    const auto value =
        std::chrono::duration_cast<std::chrono::nanoseconds>(end - begin)
            .count();
    return value > 0 ? static_cast<std::uint64_t>(value) : 0u;
  }

  bool enabled_ = false;
  Phase phase_ = Phase::Prepare;
  std::uint8_t reads_ = 0u;
  Time prepare_begin_{};
  Time work_begin_{};
  Time finish_begin_{};
  Time end_{};
};

} // namespace rund::replay::detail::scope

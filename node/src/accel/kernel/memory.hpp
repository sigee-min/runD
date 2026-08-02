#pragma once

#include <algorithm>
#include <atomic>
#include <cstdint>
#include <rund/counter.hpp>

namespace rund::node::accel::detail {

struct PreparedMemory final {
  std::uint64_t current{};
  std::uint64_t peak{};
  std::uint64_t cumulative{};
  std::uint64_t reused{};
  std::uint64_t budget{};
};

struct PreparedPipelineMemory final {
  PreparedMemory host{};
  PreparedMemory device{};
  PreparedMemory staging{};
};

inline void accumulate_memory(PreparedMemory &total,
                              const PreparedMemory value) noexcept {
  ::rund::detail::counter::Accumulate(total.current, value.current);
  ::rund::detail::counter::Accumulate(total.peak, value.peak);
  ::rund::detail::counter::Accumulate(total.cumulative, value.cumulative);
  ::rund::detail::counter::Accumulate(total.reused, value.reused);
  total.budget = std::max(total.budget, value.budget);
}

// Compose retained owners whose cold preparation phases cannot overlap. Every
// final current owner coexists, while only the largest owner-local transient
// excess over current can contribute to the aggregate preparation peak.
inline void accumulate_serial_memory(PreparedMemory &total,
                                     const PreparedMemory value) noexcept {
  const std::uint64_t total_transient =
      total.peak > total.current ? total.peak - total.current : 0u;
  const std::uint64_t value_transient =
      value.peak > value.current ? value.peak - value.current : 0u;
  accumulate_memory(total, value);
  total.peak = ::rund::detail::counter::SaturatingAdd(
      total.current, std::max(total_transient, value_transient));
}

class PreparedMemoryMeter final {
public:
  void add(const PreparedMemory value) noexcept {
    lock();
    const std::uint64_t previous = current_;
    current_ = ::rund::detail::counter::SaturatingAdd(current_, value.current);
    const std::uint64_t event_peak = std::max(value.current, value.peak);
    peak_ = std::max(
        peak_, ::rund::detail::counter::SaturatingAdd(previous, event_peak));
    cumulative_ =
        ::rund::detail::counter::SaturatingAdd(cumulative_, value.cumulative);
    reused_ = ::rund::detail::counter::SaturatingAdd(reused_, value.reused);
    budget_ = std::max(budget_, value.budget);
    unlock();
  }

  [[nodiscard]] PreparedMemory read() const noexcept {
    lock();
    const PreparedMemory result{.current = current_,
                                .peak = peak_,
                                .cumulative = cumulative_,
                                .reused = reused_,
                                .budget = budget_};
    unlock();
    return result;
  }

private:
  void lock() const noexcept {
    while (gate_.test_and_set(std::memory_order_acquire)) {
    }
  }

  void unlock() const noexcept { gate_.clear(std::memory_order_release); }

  mutable std::atomic_flag gate_{};
  std::uint64_t current_{};
  std::uint64_t peak_{};
  std::uint64_t cumulative_{};
  std::uint64_t reused_{};
  std::uint64_t budget_{};
};

} // namespace rund::node::accel::detail

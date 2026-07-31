#pragma once

#include <rund/host.hpp>
#include <rund/session.hpp>
#include <rund/task.hpp>

#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <limits>
#include <span>
#include <string_view>
#include <thread>
#include <utility>
#include <vector>

#include <sys/resource.h>
#include <unistd.h>

#if defined(__APPLE__)
#include <mach/mach.h>
#include <malloc/malloc.h>
#endif


namespace rund::measure::scheduler {

using Clock = std::chrono::steady_clock;
using Ns = std::chrono::nanoseconds;

constexpr std::size_t kWarmRounds = 7u;

struct Measure final {
  bool ok = false;
  std::string_view reason = "measure_not_started";
  double cold_ns = 0.0;
  double warm_ns = 0.0;
  std::uint64_t ops = 0u;
  rund::task::Stats stats{};
};

struct ScaleMeasure final {
  Measure total{};
  double cold_admit_ns = 0.0;
  double warm_admit_ns = 0.0;
  double cold_drain_ns = 0.0;
  double warm_drain_ns = 0.0;
  std::uint32_t payload_ops = 0u;
};

struct Pipe final {
  int read_fd = -1;
  int write_fd = -1;

  Pipe() noexcept {
    int fds[2]{-1, -1};
    if (::pipe(fds) == 0) {
      read_fd = fds[0];
      write_fd = fds[1];
    }
  }

  ~Pipe() {
    if (read_fd >= 0) {
      (void)::close(read_fd);
    }
    if (write_fd >= 0) {
      (void)::close(write_fd);
    }
  }

  Pipe(const Pipe &) = delete;
  Pipe &operator=(const Pipe &) = delete;
  [[nodiscard]] explicit operator bool() const noexcept {
    return read_fd >= 0 && write_fd >= 0;
  }
};


template <class Round>
[[nodiscard]] Measure RunMeasure(const rund::SessionConfig &config,
                                 const std::uint64_t ops, Round &&round) {
  Measure measured{};
  measured.ops = ops;
  std::vector<double> warm{};
  warm.reserve(kWarmRounds);
  bool rounds_ok = true;
  const rund::Session::Result report = rund::run(config, [&] {
    const auto run_once = [&]() {
      const auto begin = Clock::now();
      const bool ok = round();
      const auto end = Clock::now();
      rounds_ok = rounds_ok && ok;
      return static_cast<double>(
          std::chrono::duration_cast<Ns>(end - begin).count());
    };
    measured.cold_ns = run_once();
    for (std::size_t index = 0u; index < kWarmRounds; ++index) {
      warm.push_back(run_once());
    }
  });
  measured.stats = report.tasks();
  measured.ok = report && rounds_ok;
  measured.reason = report ? (rounds_ok ? std::string_view{"ok"}
                                        : std::string_view{"round_failed"})
                           : report.error();
  if (!warm.empty()) {
    std::sort(warm.begin(), warm.end());
    measured.warm_ns = warm[warm.size() / 2u];
  }
  return measured;
}


[[nodiscard]] std::uint64_t RssBytes() noexcept;
[[nodiscard]] std::uint64_t HeapBytes() noexcept;
[[nodiscard]] rund::SessionConfig Config(std::uint32_t capacity, std::uint32_t task_workers = 1u);
[[nodiscard]] bool Joined(std::span<const rund::task::Handle> tasks);
rund::task::Task<void> YieldOnce(std::atomic<std::uint64_t> *done);
rund::task::Task<void> JoinTree(std::uint32_t depth, std::atomic<std::uint64_t> *leaves, std::atomic_bool *ok);
rund::task::Task<void> Ping(rund::task::channel<std::uint32_t> *pipe, std::uint32_t rounds, std::atomic_bool *ok);
rund::task::Task<void> Pong(rund::task::channel<std::uint32_t> *pipe, std::uint32_t rounds, std::atomic_bool *ok);
rund::task::Task<void> SleepLoop(std::uint32_t rounds, std::atomic_bool *ok);
rund::task::Task<void> IoRead(Pipe *data, Pipe *ack, rund::host::io::FdView data_ready, std::uint32_t rounds, std::atomic_bool *ok);
rund::task::Task<void> IoWrite(Pipe *data, Pipe *ack, rund::host::io::FdView ack_ready, std::uint32_t rounds, std::atomic_bool *ok);
rund::task::Task<void> IoBatchWait(rund::host::io::FdView fd, std::uint8_t *ok);
rund::task::Task<void> Hold(rund::task::channel<std::uint32_t> *gate, std::atomic<std::uint32_t> *started);
rund::task::Task<void> HoldGroup(rund::task::channel<std::uint32_t> *gate, std::span<rund::task::Handle> handles, std::atomic<std::uint32_t> *started, std::uint64_t *active_rss, std::uint64_t *active_heap, bool *ok, std::string_view *reason);
void Print(const char *name, const Measure &value);
void PrintScale(const ScaleMeasure &value);
[[nodiscard]] int Latency();
[[nodiscard]] int Scale(std::uint32_t tasks, std::uint32_t task_workers, std::uint32_t payload_ops);
[[nodiscard]] int Memory(std::uint32_t tasks, std::uint32_t task_workers);

} // namespace rund::measure::scheduler

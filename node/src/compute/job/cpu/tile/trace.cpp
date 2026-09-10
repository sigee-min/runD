#include "local.hpp"

#include <chrono>

namespace rund::compute::detail::cpu_tile_detail {
namespace {

[[nodiscard]] std::uint64_t trace_now_ns() noexcept {
  const auto elapsed = std::chrono::steady_clock::now().time_since_epoch();
  return static_cast<std::uint64_t>(
      std::chrono::duration_cast<std::chrono::nanoseconds>(elapsed).count());
}

} // namespace

void begin_trace_dispatch(CpuRun &run) noexcept {
  if (!run.trace_kernel_dispatches) {
    return;
  }
  run.trace_dispatch_started_ns = trace_now_ns();
  run.trace_dispatch_active = true;
}

void cancel_trace_dispatch(CpuRun &run) noexcept {
  run.trace_dispatch_started_ns = 0u;
  run.trace_dispatch_active = false;
}

void finish_trace_dispatch(CpuRun &run, const bool timestamped) noexcept {
  if (!run.trace_dispatch_active) {
    return;
  }
  const std::uint64_t started = run.trace_dispatch_started_ns;
  cancel_trace_dispatch(run);
  if (!timestamped) {
    return;
  }
  const std::uint64_t finished = trace_now_ns();
  ::rund::detail::counter::Accumulate(
      run.stats.kernel_ns, finished >= started ? finished - started : 0u);
  ::rund::detail::counter::Accumulate(run.stats.kernel_samples, 1u);
}

} // namespace rund::compute::detail::cpu_tile_detail

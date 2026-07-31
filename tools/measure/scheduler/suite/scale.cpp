#include "model.hpp"

namespace rund::measure::scheduler {

[[gnu::noinline]] std::uint64_t RunPayload(std::uint64_t value,
                                           const std::uint32_t ops) noexcept {
  for (std::uint32_t index = 0u; index < ops; ++index) {
    value ^= value >> 7u;
    value *= 0x9e3779b185ebca87ull;
    value ^= value << 11u;
  }
  return value;
}

[[nodiscard]] ScaleMeasure MeasureScale(const std::uint32_t tasks,
                                        const std::uint32_t task_workers,
                                        const std::uint32_t payload_ops) {
  ScaleMeasure measured{};
  measured.total.ops = tasks;
  measured.payload_ops = payload_ops;
  std::vector<rund::task::Handle> handles{};
  handles.reserve(tasks);
  std::vector<std::uint64_t> output(tasks, 0u);
  std::vector<double> warm_total{};
  std::vector<double> warm_admit{};
  std::vector<double> warm_drain{};
  warm_total.reserve(kWarmRounds);
  warm_admit.reserve(kWarmRounds);
  warm_drain.reserve(kWarmRounds);
  bool rounds_ok = true;
  const rund::Session::Result report =
      rund::run(Config(tasks + 16u, task_workers), [&] {
        const auto run_once = [&](double *const admit_ns,
                                  double *const drain_ns) {
          handles.clear();
          const auto begin = Clock::now();
          for (std::uint32_t index = 0u; index < tasks; ++index) {
            if (payload_ops == 0u) {
              handles.push_back(rund::task::spawn("scale-leaf", [] {}));
            } else {
              handles.push_back(
                  rund::task::spawn("scale-work", [slot = output.data() + index,
                                                   index, payload_ops] {
                    *slot = RunPayload(static_cast<std::uint64_t>(index) + 1u,
                                       payload_ops);
                  }));
            }
          }
          const auto admitted = Clock::now();
          const bool ok = Joined(handles);
          const auto drained = Clock::now();
          *admit_ns = static_cast<double>(
              std::chrono::duration_cast<Ns>(admitted - begin).count());
          *drain_ns = static_cast<double>(
              std::chrono::duration_cast<Ns>(drained - admitted).count());
          return ok;
        };
        double cold_admit = 0.0;
        double cold_drain = 0.0;
        rounds_ok = run_once(&cold_admit, &cold_drain) && rounds_ok;
        measured.cold_admit_ns = cold_admit;
        measured.cold_drain_ns = cold_drain;
        measured.total.cold_ns = cold_admit + cold_drain;
        for (std::size_t index = 0u; index < kWarmRounds; ++index) {
          double admit = 0.0;
          double drain = 0.0;
          rounds_ok = run_once(&admit, &drain) && rounds_ok;
          warm_admit.push_back(admit);
          warm_drain.push_back(drain);
          warm_total.push_back(admit + drain);
        }
      });
  measured.total.stats = report.tasks();
  measured.total.ok = report && rounds_ok;
  measured.total.reason = report
                              ? (rounds_ok ? std::string_view{"ok"}
                                           : std::string_view{"round_failed"})
                              : report.error();
  const auto median = [](std::vector<double> &values) {
    std::sort(values.begin(), values.end());
    return values.empty() ? 0.0 : values[values.size() / 2u];
  };
  measured.total.warm_ns = median(warm_total);
  measured.warm_admit_ns = median(warm_admit);
  measured.warm_drain_ns = median(warm_drain);
  return measured;
}

[[nodiscard]] int Scale(const std::uint32_t tasks,
                        const std::uint32_t task_workers,
                        const std::uint32_t payload_ops) {
  if (tasks == 0u || tasks > 100000u) {
    std::fprintf(stderr, "scale task count must be 1..100000\n");
    return 2;
  }
  const ScaleMeasure scale = MeasureScale(tasks, task_workers, payload_ops);
  std::printf("task_scale_workers=%u ", task_workers);
  PrintScale(scale);
  return scale.total.ok ? 0 : 1;
}


} // namespace rund::measure::scheduler

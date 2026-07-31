#include "model.hpp"

namespace package_device_program {

[[nodiscard]] int CheckProfile(Device &device, const Backend backend) {
  using Real = Fixed<20, 44>;
  constexpr auto scalar = [](const std::int64_t value) {
    return Real::from_raw(value << Real::fraction_bits);
  };
  constexpr std::array<Real, 4u> singular{scalar(1), scalar(2), scalar(2),
                                          scalar(4)};
  auto pass = on(device)
                  .map<Real>("installed-profile-failure-pass", singular.size(),
                             [](auto value) { return quantize<Real>(value); })
                  .compile();
  auto factor =
      on(device)
          .map<Real>("installed-profile-failure-factor", singular.size(),
                     [](auto value) { return quantize<Real>(value); })
          .matrix<2u, 2u>()
          .lu()
          .compile();
  if (!pass || !factor) {
    return 1;
  }
  const auto make = [&](const PipelineProfile mode) -> Result<Pipeline> {
    auto input = device.upload<Real>(singular);
    auto middle = device.buffer<Real>(singular.size());
    auto values = device.buffer<Real>(singular.size());
    auto pivots = device.buffer<std::uint32_t>(2u);
    auto status = device.buffer<std::uint32_t>(1u);
    if (!input)
      return Result<Pipeline>::fail(input.reason());
    if (!middle)
      return Result<Pipeline>::fail(middle.reason());
    if (!values)
      return Result<Pipeline>::fail(values.reason());
    if (!pivots)
      return Result<Pipeline>::fail(pivots.reason());
    if (!status)
      return Result<Pipeline>::fail(status.reason());
    return pipeline(device)
        .profile(mode)
        .then(*pass, read(*input), write(*middle))
        .then(*factor, read(*middle), write(*values, *pivots, *status))
        .prepare();
  };
  auto baseline = make(PipelineProfile::None);
  auto profiled = make(PipelineProfile::Steps);
  if (!baseline || !profiled) {
    return 2;
  }
  const Status baseline_failure = baseline->run();
  const Status profiled_failure = profiled->run();
  const Stats baseline_stats = baseline->stats();
  const Stats profiled_stats = profiled->stats();
  std::array<PipelineStepProfile, 2u> rows{};
  const auto profile = profiled->profile(rows);
  std::array<PipelineStepProfile, 2u> unavailable_rows{};
  const auto unavailable_profile = baseline->profile(unavailable_rows);
  const std::uint64_t referenced_bytes =
      sizeof(Real) * singular.size() * 3u + sizeof(std::uint32_t) * 3u;
  if (baseline_failure || profiled_failure ||
      baseline_failure.reason() != Reason::FactorSingular ||
      baseline_failure.reason() != profiled_failure.reason() ||
      baseline_failure.code() != profiled_failure.code() ||
      baseline_failure.error() != profiled_failure.error() ||
      baseline->fingerprint() != profiled->fingerprint() ||
      baseline->generation() != profiled->generation() ||
      baseline_stats.backend != backend || profiled_stats.backend != backend ||
      baseline_stats.command_submits != profiled_stats.command_submits ||
      baseline_stats.pipeline.verified_step_count != 1u ||
      profiled_stats.pipeline.verified_step_count != 1u ||
      baseline_stats.pipeline.failed_step_index != 1u ||
      profiled_stats.pipeline.failed_step_index != 1u ||
      !baseline->poisoned() || !profiled->poisoned() || !profile ||
      unavailable_profile ||
      unavailable_profile.reason() != Reason::ProfileUnavailable ||
      !rows[0].execution.available() || !rows[1].execution.available() ||
      !valid_profile(backend, *profile, rows,
                     std::array{pass->fingerprint(), factor->fingerprint()},
                     profiled->memory(), referenced_bytes)) {
    return 3;
  }
  const Status baseline_poison = baseline->run();
  const Status profiled_poison = profiled->run();
  return !baseline_poison && !profiled_poison &&
                 baseline_poison.reason() == Reason::PipelinePoisoned &&
                 baseline_poison.reason() == profiled_poison.reason() &&
                 baseline_poison.code() == profiled_poison.code() &&
                 baseline_poison.error() == profiled_poison.error()
             ? 0
             : 4;
}

} // namespace package_device_program

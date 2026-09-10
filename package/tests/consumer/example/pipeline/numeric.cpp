#include "local.hpp"

#include <rund/compute/pipeline.hpp>

#include <array>
#include <cstdint>

namespace rund::package_example::pipeline {

[[nodiscard]] int CheckDependentWide(rund::compute::Device &device) {
  using Real = rund::compute::Fixed<20, 44>;
  constexpr std::array<Real, 4u> position{Real::from_raw(3), Real::from_raw(6),
                                          Real::from_raw(9),
                                          Real::from_raw(12)};
  constexpr std::array<Real, 4u> velocity{Real::from_raw(2), Real::from_raw(4),
                                          Real::from_raw(6), Real::from_raw(8)};
  constexpr std::array<Real, 4u> force{Real::from_raw(1), Real::from_raw(3),
                                       Real::from_raw(5), Real::from_raw(7)};

  auto integrate = rund::compute::on(device)
                       .input<Real>(position.size())
                       .zip_input<Real>(velocity.size())
                       .zip_input<Real>(force.size())
                       .map("pipeline-installed-integrate",
                            [](auto p, auto v, auto f) {
                              return rund::compute::quantize<Real>(p + v + f);
                            })
                       .compile();
  auto advance =
      rund::compute::on(device)
          .map<Real>("pipeline-installed-advance", position.size(),
                     [](auto value) {
                       return rund::compute::quantize<Real>(value + value);
                     })
          .compile();
  auto p = device.upload<Real>(position);
  auto v = device.upload<Real>(velocity);
  auto f = device.upload<Real>(force);
  auto middle = device.buffer<Real>(position.size());
  auto output = device.buffer<Real>(position.size());
  if (!integrate || !advance || !p || !v || !f || !middle || !output) {
    return 1;
  }

  auto prepared = rund::compute::pipeline(device)
                      .profile(rund::compute::PipelineProfile::Steps)
                      .then(*integrate, rund::compute::read(*p, *v, *f),
                            rund::compute::write(*middle))
                      .then(*advance, rund::compute::read(*middle),
                            rund::compute::write(*output))
                      .prepare();
  if (!prepared) {
    return prepared.exit_code();
  }
  rund::compute::Pipeline pipeline = std::move(prepared).value();
  const auto ran = pipeline.run();
  if (!ran) {
    return ran.exit_code();
  }
  std::array<rund::compute::PipelineStepProfile, 2u> steps{};
  const auto profile = pipeline.profile(steps);
  if (!profile) {
    return profile.exit_code();
  }
  std::array<Real, position.size()> observed{};
  const auto read = pipeline.read(*output, observed);
  if (!read) {
    return read.exit_code();
  }
  return observed == std::array<Real, 4u>{Real::from_raw(12),
                                          Real::from_raw(26),
                                          Real::from_raw(40),
                                          Real::from_raw(54)} &&
                 pipeline.stats().pipeline.step_count == 2u &&
                 pipeline.stats().pipeline.resource_count == 5u &&
                 pipeline.stats().pipeline.barrier_count == 1u &&
                 pipeline.stats().command_submits == 0u &&
                 profile->written == steps.size() &&
                 profile->total == steps.size() && !profile->truncated() &&
                 steps[0].index == 0u && steps[1].index == 1u &&
                 steps[0].program == integrate->fingerprint() &&
                 steps[1].program == advance->fingerprint() &&
                 steps[0].execution.available() &&
                 steps[1].execution.available() &&
                 steps[0].timing.available() && steps[1].timing.available() &&
                 steps[0].timing.clock ==
                     rund::compute::StepClock::HostSteady &&
                 steps[1].timing.clock ==
                     rund::compute::StepClock::HostSteady &&
                 profile->memory.available() &&
                 profile->shared_memory.available() &&
                 profile->referenced_resource_bytes ==
                     sizeof(Real) * position.size() * 5u &&
                 profile->instrumentation_command_count == 0u &&
                 profile->instrumentation_byte_count == 0u &&
                 profile->observation.available()
             ? 0
             : 2;
}

} // namespace rund::package_example::pipeline

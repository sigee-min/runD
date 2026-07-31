#include "local/model.hpp"

#include <rund/compute.hpp>
#include <rund/compute/math.hpp>
#include <rund/compute/pipeline.hpp>
#include <rund/compute/session.hpp>
#include <rund/session.hpp>

#include <array>
#include <cstdint>

namespace runtime_compute_pipeline {

int Semantic(rund::Session &session, rund::compute::Device &device) {
  using namespace rund::compute;
  using Real = Fixed<20, 44>;
  constexpr auto scalar = [](const std::int64_t value) {
    return Real::from_raw(value << Real::fraction_bits);
  };
  constexpr std::array<Real, 4u> singular{scalar(1), scalar(2), scalar(2),
                                          scalar(4)};
  auto pass = on(device)
                  .map<Real>("session-pipeline-status-pass", singular.size(),
                             [](auto value) { return quantize<Real>(value); })
                  .compile();
  auto factor =
      on(device)
          .map<Real>("session-pipeline-status-factor", singular.size(),
                     [](auto value) { return quantize<Real>(value); })
          .matrix<2u, 2u>()
          .lu()
          .compile();
  auto input = device.upload<Real>(singular);
  auto middle = device.buffer<Real>(singular.size());
  auto values = device.buffer<Real>(singular.size());
  auto pivots = device.buffer<std::uint32_t>(2u);
  auto status = device.buffer<std::uint32_t>(1u);
  if (!pass || !factor || !input || !middle || !values || !pivots || !status) {
    return 1;
  }
  auto prepared =
      pipeline(device)
          .then(*pass, read(*input), write(*middle))
          .then(*factor, read(*middle), write(*values, *pivots, *status))
          .prepare();
  if (!prepared) {
    return 2;
  }
  const Completion completion = session.compute(*prepared).submit().wait();
  const Stats stats = completion.stats();
  return !completion && completion.reason() == Reason::FactorSingular &&
                 prepared->poisoned() && prepared->generation() == 0u &&
                 stats.backend == Backend::Cpu &&
                 stats.pipeline.verified_step_count == 1u &&
                 stats.pipeline.failed_step_index == 1u &&
                 stats.pipeline.status_entry_count == 1u &&
                 stats.command_submits == 0u &&
                 prepared->run().reason() == Reason::PipelinePoisoned
             ? 0
             : 3;
}

} // namespace runtime_compute_pipeline

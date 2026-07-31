#include "local/model.hpp"

#include <rund/compute.hpp>
#include <rund/compute/pipeline.hpp>
#include <rund/compute/session.hpp>
#include <rund/session.hpp>

#include <array>
#include <cstddef>
#include <cstdint>
#include <thread>
#include <vector>

namespace runtime_compute_pipeline {

int Cancellation(rund::Session &session, rund::compute::Device &device) {
  using namespace rund::compute;
  constexpr std::size_t count = 1u << 20u;
  std::vector<std::int32_t> values(count, 1);
  auto first_program =
      on(device)
          .map<std::int32_t>("session-pipeline-cancel-first", count,
                             [](auto value) { return value + 1; })
          .map("session-pipeline-cancel-first-2",
               [](auto value) { return value * 3; })
          .compile();
  auto second_program =
      on(device)
          .map<std::int32_t>("session-pipeline-cancel-second", count,
                             [](auto value) { return value + 7; })
          .map("session-pipeline-cancel-second-2",
               [](auto value) { return value * 5; })
          .compile();
  auto input = device.upload<std::int32_t>(values);
  auto middle = device.buffer<std::int32_t>(count);
  auto output = device.buffer<std::int32_t>(count);
  auto untouched = device.buffer<std::int32_t>(count);
  if (!first_program || !second_program || !input || !middle || !output ||
      !untouched) {
    return 1;
  }
  auto prepared = pipeline(device)
                      .profile(PipelineProfile::Steps)
                      .then(*first_program, read(*input), write(*middle))
                      .then(*second_program, read(*middle), write(*output))
                      .prepare();
  if (!prepared) {
    return 2;
  }
  constexpr std::size_t attempts = 64u;
  for (std::size_t attempt = 0u; attempt < attempts; ++attempt) {
    auto submission = session.compute(*prepared).submit();
    Poll observed = submission.poll();
    for (std::size_t spin = 0u;
         spin < 1000000u && !observed.backend_submitted && !observed.completed;
         ++spin) {
      std::this_thread::yield();
      observed = submission.poll();
    }
    if (!observed.backend_submitted) {
      return 3;
    }
    if (observed.completed) {
      if (!submission.wait()) {
        return 3;
      }
      continue;
    }

    std::array<PipelineStepProfile, 2u> rows{};
    const auto busy_profile = prepared->profile(rows);
    if (busy_profile) {
      if (!submission.wait()) {
        return 6;
      }
      continue;
    }
    if (busy_profile.reason() != Reason::ProfileBusy) {
      return 6;
    }

    const Status cancelled = submission.cancel();
    const Completion completion = submission.wait();
    if (!cancelled) {
      if (cancelled.reason() == Reason::AlreadyCompleted && completion) {
        continue;
      }
      return 4;
    }
    if (completion || completion.reason() != Reason::Cancelled ||
        !prepared->poisoned() ||
        prepared->run().reason() != Reason::PipelinePoisoned ||
        prepared->stats().pipeline.failed_step_index ==
            PipelineStats::no_failed_step) {
      return 4;
    }
    const auto profile = prepared->profile(rows);
    const std::size_t verified = static_cast<std::size_t>(
        completion.stats().pipeline.verified_step_count);
    if (!profile || verified > rows.size()) {
      return 7;
    }
    for (std::size_t index = 0u; index < rows.size(); ++index) {
      const bool work = rows[index].execution.available();
      const bool timing = rows[index].timing.available();
      if (work != timing || (index < verified && !work) ||
          (index > verified && work)) {
        return 8;
      }
    }
    std::vector<std::int32_t> unavailable(count);
    const auto changed = prepared->read(*middle, unavailable);
    const auto later = prepared->read(*output, unavailable);
    const auto reusable = first_program->run(*output, *untouched);
    const std::uint64_t failed =
        completion.stats().pipeline.failed_step_index;
    const bool first_step =
        failed == 0u && !later && later.reason() == Reason::PipelinePoisoned &&
        reusable;
    const bool second_step =
        failed == 1u && !later && later.reason() == Reason::BufferPoisoned &&
        !reusable && reusable.reason() == Reason::BufferPoisoned;
    return !changed && changed.reason() == Reason::BufferPoisoned &&
                   (first_step || second_step)
               ? 0
               : 5;
  }
  return 9;
}

} // namespace runtime_compute_pipeline

#include "local/model.hpp"

#include "../../support.hpp"

#include <rund/compute.hpp>
#include <rund/compute/pipeline.hpp>
#include <rund/compute/session.hpp>
#include <rund/session.hpp>
#include <rund/task/api.hpp>
#include <rund/task/await.hpp>

#include <array>
#include <cstddef>
#include <cstdint>
#include <span>
#include <thread>
#include <vector>

namespace runtime_compute_pipeline {
namespace {

rund::task::Task<void> Await(rund::Session &session,
                             rund::compute::Pipeline &pipeline, bool &completed,
                             rund::compute::Reason &reason) {
  const rund::compute::Completion completion =
      co_await session.compute(pipeline);
  completed = static_cast<bool>(completion);
  reason = completion.reason();
  co_return;
}

} // namespace

int Lifecycle(rund::Session &session, rund::compute::Device &device) {
  using namespace rund::compute;
  constexpr std::array<std::int32_t, 4u> input_values{1, 2, 3, 4};
  auto first =
      on(device)
          .map<std::int32_t>("session-pipeline-double", input_values.size(),
                             [](auto value) { return value * 2; })
          .compile();
  auto second =
      on(device)
          .map<std::int32_t>("session-pipeline-advance", input_values.size(),
                             [](auto value) { return value + 3; })
          .compile();
  auto input = device.upload<std::int32_t>(input_values);
  auto middle = device.buffer<std::int32_t>(input_values.size());
  auto output = device.buffer<std::int32_t>(input_values.size());
  auto await_middle = device.buffer<std::int32_t>(input_values.size());
  auto await_output = device.buffer<std::int32_t>(input_values.size());
  if (!first || !second || !input || !middle || !output || !await_middle ||
      !await_output) {
    return 1;
  }
  auto prepared = pipeline(device)
                      .profile(PipelineProfile::Steps)
                      .then(*first, read(*input), write(*middle))
                      .then(*second, read(*middle), write(*output))
                      .prepare();
  auto awaited = pipeline(device)
                     .profile(PipelineProfile::Steps)
                     .then(*first, read(*input), write(*await_middle))
                     .then(*second, read(*await_middle), write(*await_output))
                     .prepare();
  if (!prepared || !awaited) {
    return 2;
  }

  auto submission = session.compute(*prepared).submit();
  const Poll admitted = submission.poll();
  if (!admitted.submitted || admitted.reason() != Reason::Ok) {
    return 3;
  }
  const Completion completion = submission.wait();
  const Stats stats = completion.stats();
  if (!completion || stats.backend != Backend::Cpu ||
      stats.pipeline.step_count != 2u || stats.pipeline.resource_count != 3u ||
      stats.pipeline.barrier_count != 1u ||
      stats.pipeline.verified_step_count != 2u ||
      stats.pipeline.failed_step_index != PipelineStats::no_failed_step ||
      stats.command_submits != 0u || prepared->generation() != 1u) {
    return 4;
  }
  const Poll terminal = submission.poll();
  if (!terminal.submitted || !terminal.completed ||
      terminal.reason() != Reason::Ok) {
    return 5;
  }
  std::array<std::int32_t, input_values.size()> values{};
  if (!ReadExact(*prepared, *output, std::span<std::int32_t>{values}) ||
      values != std::array<std::int32_t, 4u>{5, 7, 9, 11}) {
    return 6;
  }
  std::array<PipelineStepProfile, 2u> rows{};
  const auto profile = prepared->profile(rows);
  if (!profile || profile->written != rows.size() ||
      profile->total != rows.size() || profile->truncated() ||
      rows[0].index != 0u || rows[1].index != 1u ||
      !rows[0].execution.available() || !rows[1].execution.available() ||
      !rows[0].timing.available() || !rows[1].timing.available() ||
      rows[0].timing.clock != StepClock::HostSteady ||
      rows[1].timing.clock != StepClock::HostSteady ||
      rows[0].timing.relation != StepTimingRelation::Exclusive ||
      rows[1].timing.relation != StepTimingRelation::Exclusive) {
    return 8;
  }

  bool await_completed = false;
  bool joined = false;
  Reason await_reason = Reason::TaskInvalid;
  const rund::Session::Result scope = session.scope([&] {
    const rund::task::Handle handle = rund::task::spawn(
        "pipeline-await",
        Await(session, *awaited, await_completed, await_reason));
    joined = static_cast<bool>(rund::task::join(handle));
  });
  if (!scope || !joined || !await_completed || await_reason != Reason::Ok ||
      !ReadExact(*awaited, *await_output, std::span<std::int32_t>{values}) ||
      values != std::array<std::int32_t, 4u>{5, 7, 9, 11}) {
    return 7;
  }
  const auto await_profile = awaited->profile(rows);
  if (!await_profile || !rows[0].execution.available() ||
      !rows[1].execution.available() || !rows[0].timing.available() ||
      !rows[1].timing.available()) {
    return 9;
  }
  return 0;
}

int Close(rund::compute::Device &device) {
  using namespace rund::compute;
  constexpr std::size_t count = 1u << 20u;
  std::vector<std::int32_t> values(count, 3);
  auto program = on(device)
                     .map<std::int32_t>("session-pipeline-close", count,
                                        [](auto value) { return value * 7; })
                     .map("session-pipeline-close-second",
                          [](auto value) { return value + 11; })
                     .compile();
  auto input = device.upload<std::int32_t>(values);
  auto output = device.buffer<std::int32_t>(count);
  if (!program || !input || !output) {
    return 1;
  }
  auto prepared =
      pipeline(device).then(*program, read(*input), write(*output)).prepare();
  if (!prepared) {
    return 2;
  }
  constexpr std::size_t attempts = 64u;
  for (std::size_t attempt = 0u; attempt < attempts; ++attempt) {
    rund::Session session{};
    if (!session.open(rund::node::test_contract::Options())) {
      return 3;
    }
    auto submission = session.compute(*prepared).submit();
    Poll observed = submission.poll();
    for (std::size_t spin = 0u;
         spin < 1000000u && !observed.backend_submitted && !observed.completed;
         ++spin) {
      std::this_thread::yield();
      observed = submission.poll();
    }
    if (!observed.backend_submitted) {
      return 4;
    }
    if (observed.completed) {
      if (!submission.wait() || !session.close()) {
        return 5;
      }
      continue;
    }
    if (!session.close()) {
      return 5;
    }
    const Completion completion = submission.wait();
    if (completion) {
      if (session.snapshot().state != rund::SessionState::Stopped) {
        return 6;
      }
      continue;
    }
    if (completion.reason() != Reason::Cancelled || !prepared->poisoned() ||
        session.snapshot().state != rund::SessionState::Stopped) {
      return 6;
    }
    const Completion stopped = session.compute(*prepared).submit().wait();
    return !stopped && stopped.reason() == Reason::RuntimeMissing ? 0 : 7;
  }
  return 8;
}

} // namespace runtime_compute_pipeline

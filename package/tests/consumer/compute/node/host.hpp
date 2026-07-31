#pragma once

#include <rund/compute.hpp>
#include <rund/compute/session.hpp>
#include <rund/task.hpp>

#include <array>
#include <cstdint>
#include <vector>

namespace package_compute {
inline rund::task::Task<void>
AwaitCompute(rund::Session &session,
             rund::compute::Job<std::int32_t(std::int32_t)> &job, bool &ran,
             int &failure) {
  const auto result = co_await session.compute(job);
  if (!result) {
    failure = result.exit_code();
    co_return;
  }
  ran = result && result.stats().download_events == 0u;
}
inline int NodeHost() {
  const std::array<std::int32_t, 4> input{1, 2, 3, 4};
  auto program =
      rund::compute::on(rund::compute::Target::cpu(2u))
          .map<std::int32_t>("node-host", input.size(),
                             [](auto value) { return value * 2 + 5; })
          .compile();
  if (!program) {
    return program.exit_code();
  }
  auto job = program->resident(input);
  auto awaited_job = program->resident(input);
  if (!job) {
    return job.exit_code();
  }
  if (!awaited_job) {
    return awaited_job.exit_code();
  }

  bool ran = false;
  bool awaited = false;
  int failure = 0;
  auto hosted = rund::run({.workers = 2u}, [&](rund::Session &session) {
    const auto result = session.compute(*job).submit().wait();
    if (!result) {
      failure = result.exit_code();
      return;
    }
    ran = result.stats().download_events == 0u;
    const auto task = rund::task::spawn(
        "package-compute-await",
        AwaitCompute(session, *awaited_job, awaited, failure));
    const auto joined = rund::task::join(task);
    if (!joined) {
      failure = joined.exit_code();
      return;
    }
  });
  if (!hosted) {
    return hosted.exit_code();
  }
  if (failure != 0) {
    return failure;
  }
  if (!ran || !awaited) {
    return 2;
  }
  auto output = job->read();
  auto awaited_output = awaited_job->read();
  if (!output) {
    return output.exit_code();
  }
  if (!awaited_output) {
    return awaited_output.exit_code();
  }
  if (*output != std::vector<std::int32_t>{7, 9, 11, 13} ||
      *awaited_output != std::vector<std::int32_t>{7, 9, 11, 13}) {
    return 2;
  }
  return 0;
}

} // namespace package_compute

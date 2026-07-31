#pragma once

#include "../../support.hpp"

#include <rund/compute.hpp>
#include <rund/compute/session/submission.hpp>

#include <algorithm>
#include <cstdint>
#include <type_traits>
#include <vector>

namespace compute_host_test {

static_assert(!std::is_copy_constructible_v<rund::compute::Submission>);
static_assert(std::is_move_constructible_v<rund::compute::Submission>);
static_assert(!std::is_copy_constructible_v<rund::compute::Request>);
static_assert(std::is_move_constructible_v<rund::compute::Request>);

[[nodiscard]] inline bool
OutputMatches(rund::compute::Job<std::int32_t(std::int32_t)> &job,
              const std::vector<std::int32_t> &input) {
  auto output = job.read();
  return output && output->size() == input.size() &&
         std::equal(
             output->begin(), output->end(), input.begin(),
             [](const auto out, const auto in) { return out == in * 2 + 5; });
}

[[nodiscard]] inline bool HasComputeTrace(const rund::Session &session) {
  const rund::Trace trace = session.trace();
  return rund::node::test_contract::Saw(trace,
                                        rund::TraceEvent::ComputeSubmitted) &&
         rund::node::test_contract::Saw(trace,
                                        rund::TraceEvent::ComputeAdmitted) &&
         rund::node::test_contract::Saw(
             trace, rund::TraceEvent::ComputeDispatchStarted) &&
         rund::node::test_contract::Saw(
             trace, rund::TraceEvent::ComputeBackendSubmitted) &&
         rund::node::test_contract::Saw(trace,
                                        rund::TraceEvent::ComputeCompleted) &&
         rund::node::test_contract::Saw(trace,
                                        rund::TraceEvent::TelemetryEmitted);
}

} // namespace compute_host_test

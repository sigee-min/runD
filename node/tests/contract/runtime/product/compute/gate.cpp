#include "../support.hpp"

#include <node/runtime/compute/access.hpp>
#include <rund/compute.hpp>
#include <rund/compute/session.hpp>

#include "src/compute/job/state.hpp"

#include <array>
#include <cstdint>
#include <string_view>

namespace rund::node::test_contract {

int CheckComputeJobGate(::rund::Session &session,
                        const std::array<std::int32_t, 4> &input) {
  auto program =
      compute::on(compute::Target::cpu(2u))
          .map<std::int32_t>("node-host-gate", input.size(),
                             [](auto value) { return value * 2 + 5; })
          .compile();
  if (!program) {
    return 1;
  }
  auto job = program->resident(input);
  if (!job) {
    return 2;
  }
  const auto state = compute::detail::JobAccess::state(*job);
  if (!compute::detail::queue_job(state)) {
    return 3;
  }
  const auto queued_read = job->read();
  if (queued_read ||
      queued_read.error() != std::string_view{"compute_job_running"}) {
    return 4;
  }
  const auto queued_profile = job->profile();
  if (queued_profile || queued_profile.code() != compute::Code::Execution ||
      queued_profile.error() != std::string_view{"compute_profile_busy"}) {
    return 5;
  }
  const std::array<std::int32_t, 4> replacement{5, 6, 7, 8};
  const auto queued_write = job->write(replacement);
  if (queued_write ||
      queued_write.error() != std::string_view{"compute_job_busy"}) {
    return 6;
  }
  const auto duplicate = session.compute(*job).submit().wait();
  if (duplicate || duplicate.error() != std::string_view{"compute_job_busy"}) {
    return 7;
  }
  const auto cancelled = compute::detail::cancel_job(state);
  if (cancelled || cancelled.error() != std::string_view{"compute_cancelled"}) {
    return 8;
  }
  const auto cancelled_read = job->read();
  if (cancelled_read ||
      cancelled_read.error() != std::string_view{"compute_cancelled"}) {
    return 9;
  }
  const auto rerun = session.compute(*job).submit().wait();
  if (!rerun) {
    return 10;
  }
  if (!job->profile()) {
    return 11;
  }

  compute::Submission retained{};
  {
    auto retained_job = program->resident(input);
    if (!retained_job) {
      return 12;
    }
    retained = session.compute(*retained_job).submit();
  }
  return retained.wait() ? 0 : 13;
}

} // namespace rund::node::test_contract

#include "../support.hpp"

#include <rund/compute.hpp>
#include <rund/compute/session.hpp>
#include <rund/task/api.hpp>
#include <rund/task/await.hpp>

#include <array>
#include <cstdint>
#include <utility>
#include <vector>

namespace rund::node::test_contract {
namespace {

rund::task::Task<void> AwaitCompute(::rund::Session &session,
                        compute::Job<std::int32_t(std::int32_t)> &job,
                        bool &completed) {
  const compute::Completion result = co_await session.compute(job);
  completed = static_cast<bool>(result);
  co_return;
}

rund::task::Task<void> DiscardCompute(::rund::Session &session,
                          compute::Job<std::int32_t(std::int32_t)> &job) {
  auto request = session.compute(job);
  (void)request;
  (void)co_await rund::task::yield();
}

rund::task::Task<void> AwaitInvalid(compute::Reason &reason) {
  compute::Request request{};
  const compute::Completion result = co_await std::move(request);
  reason = result.reason();
}

} // namespace

int CheckComputeAwait(::rund::Session &session) {
  constexpr std::array<std::int32_t, 4> input{1, 2, 3, 4};
  auto program =
      compute::on(compute::Target::cpu(2u))
          .map<std::int32_t>("node-host-await", input.size(),
                             [](auto value) { return value * 3 + 1; })
          .compile();
  if (!program) {
    return 1;
  }
  auto job = program->resident(input);
  auto discarded_job = program->resident(input);
  if (!job || !discarded_job) {
    return 2;
  }

  bool discarded_joined = false;
  const rund::Session::Result discarded_scope = session.scope([&] {
    const rund::task::Handle task =
        rund::task::spawn("compute-discard", DiscardCompute(session, *discarded_job));
    discarded_joined = static_cast<bool>(rund::task::join(task));
  });
  if (!discarded_scope || !discarded_joined || discarded_job->read() ||
      discarded_job->stats().dispatches != 0u) {
    return 3;
  }

  bool completed = false;
  bool joined = false;
  bool invalid_joined = false;
  compute::Reason invalid_reason = compute::Reason::Ok;
  const rund::Session::Result scope = session.scope([&] {
    const rund::task::Handle task =
        rund::task::spawn("compute-await", AwaitCompute(session, *job, completed));
    joined = static_cast<bool>(rund::task::join(task));
    const rund::task::Handle invalid =
        rund::task::spawn("compute-invalid", AwaitInvalid(invalid_reason));
    invalid_joined = static_cast<bool>(rund::task::join(invalid));
  });
  if (!scope || !joined || !completed || !invalid_joined ||
      invalid_reason != compute::Reason::TaskInvalid) {
    return 4;
  }
  const auto output = job->read();
  return output && *output == std::vector<std::int32_t>{4, 7, 10, 13} ? 0 : 5;
}

} // namespace rund::node::test_contract

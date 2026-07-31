#include "test/assert.hpp"

#include "src/runtime/compute/state.hpp"
#include "src/runtime/compute/terminal.hpp"
#include "src/runtime/compute/operation.hpp"

#include <node/runtime/compute/access.hpp>
#include <rund/compute.hpp>

#include <array>
#include <atomic>
#include <cstdint>
#include <string_view>
#include <type_traits>

namespace {

using ComputeTelemetryEmit = void (*)(void *, const rund::compute::Status &,
                                      const rund::compute::Stats &) noexcept;

static_assert(std::is_same_v<rund::node::runtime_detail::ComputeHostState::Emit,
                             ComputeTelemetryEmit>);

} // namespace

int RunRuntimeComputeTerminalContract() {
  using rund::node::compute_detail::CancelClaim;
  using rund::node::compute_detail::ClaimFinish;
  using rund::node::compute_detail::FinishClaim;
  using rund::node::compute_detail::FinishFailure;
  using rund::node::compute_detail::MarkComplete;
  using rund::node::compute_detail::RequestCancel;
  using rund::node::compute_detail::TaskState;
  using rund::node::compute_detail::TerminalPhase;

  std::atomic phase{TerminalPhase::Open};
  TEST_ASSERT(RequestCancel(phase) == CancelClaim::Accept);
  TEST_ASSERT(RequestCancel(phase) == CancelClaim::Cancelled);
  TEST_ASSERT(ClaimFinish(phase) == FinishClaim::Cancel);
  MarkComplete(phase);
  TEST_ASSERT(RequestCancel(phase) == CancelClaim::Closed);
  TEST_ASSERT(ClaimFinish(phase) == FinishClaim::Closed);

  std::atomic finishing{TerminalPhase::Open};
  TEST_ASSERT(ClaimFinish(finishing) == FinishClaim::Finish);
  TEST_ASSERT(RequestCancel(finishing) == CancelClaim::Closed);
  TEST_ASSERT(ClaimFinish(finishing) == FinishClaim::Closed);

  constexpr std::array<std::int32_t, 4> input{1, 2, 3, 4};
  auto program = rund::compute::on(rund::compute::Target::cpu(1u))
                     .map<std::int32_t>("terminal-cancel", input.size(),
                                        [](auto value) { return value * 2; })
                     .compile();
  TEST_ASSERT(program);
  auto job = program->resident(input);
  TEST_ASSERT(job);

  TaskState task{};
  const auto job_state = rund::compute::detail::JobAccess::state(*job);
  task.operation = rund::node::compute_detail::make_job(job_state);
  TEST_ASSERT(rund::compute::detail::queue_job(job_state));
  TEST_ASSERT(RequestCancel(task.terminal_phase) == CancelClaim::Accept);
  const rund::compute::Status cancelled = FinishFailure(
      task,
      rund::compute::Status::fail(
          rund::compute::Reason::PrimitiveBackendFailed));
  TEST_ASSERT(!cancelled);
  TEST_ASSERT(cancelled.error() == std::string_view{"compute_cancelled"});
  const auto output = job->read();
  TEST_ASSERT(!output);
  TEST_ASSERT(output.error() == std::string_view{"compute_cancelled"});
  return 0;
}

#include "test/assert.hpp"

#include "src/compute/job/cpu/model.hpp"
#include "src/runtime/compute/operation.hpp"
#include "src/runtime/compute/state.hpp"
#include "src/runtime/compute/terminal.hpp"

#include <node/runtime/compute/access.hpp>
#include <rund/compute.hpp>

#include <array>
#include <atomic>
#include <cstdint>
#include <memory>
#include <string_view>
#include <type_traits>

namespace {

using ComputeTelemetryEmit = void (*)(void *, const rund::compute::Status &,
                                      const rund::compute::Stats &) noexcept;

static_assert(std::is_same_v<rund::node::runtime_detail::ComputeHostState::Emit,
                             ComputeTelemetryEmit>);

using rund::compute::detail::CpuStepDisposition;
using rund::compute::detail::CpuStepProgress;
using rund::node::compute_detail::Advance;
using rund::node::compute_detail::AdvanceDisposition;
using rund::node::compute_detail::Dispatch;
using rund::node::compute_detail::DispatchDisposition;

static_assert(!std::is_default_constructible_v<Dispatch>);
static_assert(!std::is_aggregate_v<Dispatch>);
static_assert(std::is_trivially_copyable_v<Dispatch>);
static_assert(!std::is_default_constructible_v<Advance>);
static_assert(!std::is_aggregate_v<Advance>);
static_assert(std::is_trivially_copyable_v<Advance>);

constexpr Dispatch FailedDispatch = Dispatch::failed(
    rund::compute::Status::fail(rund::compute::Reason::RuntimeMissing));
constexpr Dispatch AcceptedDispatch = Dispatch::accepted_without_backend();
constexpr Dispatch SubmittedDispatch = Dispatch::backend_submitted();
static_assert(FailedDispatch.disposition() == DispatchDisposition::Failed);
static_assert(FailedDispatch.status().reason() ==
              rund::compute::Reason::RuntimeMissing);
static_assert(AcceptedDispatch.disposition() ==
              DispatchDisposition::AcceptedNoBackend);
static_assert(AcceptedDispatch.status());
static_assert(SubmittedDispatch.disposition() ==
              DispatchDisposition::BackendSubmitted);
static_assert(SubmittedDispatch.status());
static_assert(
    Dispatch::failed(rund::compute::Status::success()).status().reason() ==
    rund::compute::Reason::CompletionInvalid);

constexpr Advance FailedAdvance = Advance::failed(
    rund::compute::Status::fail(rund::compute::Reason::RuntimeMissing));
constexpr Advance PendingAdvance = Advance::pending();
constexpr Advance SubmittedAdvance = Advance::backend_submitted();
constexpr Advance CompleteAdvance = Advance::complete();
static_assert(FailedAdvance.disposition() == AdvanceDisposition::Failed);
static_assert(FailedAdvance.status().reason() ==
              rund::compute::Reason::RuntimeMissing);
static_assert(PendingAdvance.disposition() == AdvanceDisposition::Pending);
static_assert(PendingAdvance.status());
static_assert(SubmittedAdvance.disposition() ==
              AdvanceDisposition::BackendSubmitted);
static_assert(SubmittedAdvance.status());
static_assert(CompleteAdvance.disposition() == AdvanceDisposition::Complete);
static_assert(CompleteAdvance.status());
static_assert(
    Advance::failed(rund::compute::Status::success()).status().reason() ==
    rund::compute::Reason::CompletionInvalid);

static_assert(!std::is_default_constructible_v<CpuStepProgress>);
static_assert(!std::is_aggregate_v<CpuStepProgress>);
static_assert(std::is_trivially_copyable_v<CpuStepProgress>);

constexpr CpuStepProgress FailedCpuStep = CpuStepProgress::failed(
    rund::compute::Status::fail(rund::compute::Reason::Cancelled));
constexpr CpuStepProgress PendingCpuStep = CpuStepProgress::pending();
constexpr CpuStepProgress CompleteCpuStep = CpuStepProgress::complete();
static_assert(FailedCpuStep.disposition() == CpuStepDisposition::Failed);
static_assert(FailedCpuStep.status().reason() ==
              rund::compute::Reason::Cancelled);
static_assert(PendingCpuStep.disposition() == CpuStepDisposition::Pending);
static_assert(PendingCpuStep.status());
static_assert(CompleteCpuStep.disposition() == CpuStepDisposition::Complete);
static_assert(CompleteCpuStep.status());
static_assert(CpuStepProgress::failed(rund::compute::Status::success())
                  .status()
                  .reason() == rund::compute::Reason::CpuStepInvalid);

std::atomic<std::uint32_t> configured_abort_cancels{0u};
std::atomic<std::uint32_t> configured_abort_retires{0u};

void CountConfiguredAbortCancel(
    const std::shared_ptr<rund::node::runtime_detail::ComputeHostState> &)
    noexcept {
  configured_abort_cancels.fetch_add(1u, std::memory_order_relaxed);
}

void CountConfiguredAbortRetire(
    const std::shared_ptr<rund::node::runtime_detail::ComputeHostState> &)
    noexcept {
  configured_abort_retires.fetch_add(1u, std::memory_order_relaxed);
}

} // namespace

int RunRuntimeComputeTerminalContract() {
  using rund::node::runtime_detail::ComputeHostAdmission;
  using rund::node::runtime_detail::ComputeHostCloseClaim;
  using rund::node::runtime_detail::ComputeHostLifecycle;
  using rund::node::runtime_detail::ComputeHostPhase;
  using rund::node::compute_detail::CancelClaim;
  using rund::node::compute_detail::ClaimFinish;
  using rund::node::compute_detail::FinishClaim;
  using rund::node::compute_detail::FinishFailure;
  using rund::node::compute_detail::MarkComplete;
  using rund::node::compute_detail::RequestCancel;
  using rund::node::compute_detail::TaskRetirement;
  using rund::node::compute_detail::TaskRetirementClaim;
  using rund::node::compute_detail::TaskRetirementPhase;
  using rund::node::compute_detail::TaskState;
  using rund::node::compute_detail::TerminalPhase;

  ComputeHostLifecycle host_lifecycle{};
  TEST_ASSERT(host_lifecycle.phase() == ComputeHostPhase::Constructing);
  TEST_ASSERT(host_lifecycle.admission() == ComputeHostAdmission::Offline);
  TEST_ASSERT(host_lifecycle.bindable());
  TEST_ASSERT(!host_lifecycle.closed());
  TEST_ASSERT(!host_lifecycle.start());
  TEST_ASSERT(!host_lifecycle.stop());
  TEST_ASSERT(host_lifecycle.configure());
  TEST_ASSERT(!host_lifecycle.configure());
  TEST_ASSERT(host_lifecycle.phase() == ComputeHostPhase::Configured);
  TEST_ASSERT(host_lifecycle.admission() == ComputeHostAdmission::Standby);
  TEST_ASSERT(host_lifecycle.start());
  TEST_ASSERT(host_lifecycle.phase() == ComputeHostPhase::Running);
  TEST_ASSERT(host_lifecycle.admission() == ComputeHostAdmission::Open);
  TEST_ASSERT(host_lifecycle.stop());
  TEST_ASSERT(!host_lifecycle.stop());
  TEST_ASSERT(host_lifecycle.phase() == ComputeHostPhase::Draining);
  TEST_ASSERT(host_lifecycle.admission() == ComputeHostAdmission::Draining);
  TEST_ASSERT(host_lifecycle.claim_close() == ComputeHostCloseClaim::Own);
  TEST_ASSERT(host_lifecycle.phase() == ComputeHostPhase::Closing);
  TEST_ASSERT(host_lifecycle.claim_close() == ComputeHostCloseClaim::Wait);
  TEST_ASSERT(!host_lifecycle.bindable());
  host_lifecycle.begin_retirement();
  TEST_ASSERT(host_lifecycle.phase() == ComputeHostPhase::Retiring);
  TEST_ASSERT(host_lifecycle.admission() == ComputeHostAdmission::Offline);
  host_lifecycle.publish_closed();
  TEST_ASSERT(host_lifecycle.phase() == ComputeHostPhase::Closed);
  TEST_ASSERT(host_lifecycle.closed());
  TEST_ASSERT(host_lifecycle.claim_close() == ComputeHostCloseClaim::Closed);

  configured_abort_cancels.store(0u, std::memory_order_relaxed);
  configured_abort_retires.store(0u, std::memory_order_relaxed);
  auto configured_abort =
      std::make_shared<rund::node::runtime_detail::ComputeHostState>();
  TEST_ASSERT(configured_abort->lifecycle.configure());
  rund::node::runtime_detail::BindLifecycle(
      configured_abort, CountConfiguredAbortCancel,
      CountConfiguredAbortRetire);
  rund::node::runtime_detail::CloseHost(configured_abort);
  TEST_ASSERT(configured_abort->lifecycle.phase() == ComputeHostPhase::Closed);
  TEST_ASSERT(configured_abort_cancels.load(std::memory_order_relaxed) == 0u);
  TEST_ASSERT(configured_abort_retires.load(std::memory_order_relaxed) == 1u);
  rund::node::runtime_detail::CloseHost(configured_abort);
  TEST_ASSERT(configured_abort_retires.load(std::memory_order_relaxed) == 1u);

  TaskRetirement retirement{};
  TEST_ASSERT(retirement.phase() == TaskRetirementPhase::Live);
  TEST_ASSERT(retirement.claim(true) == TaskRetirementClaim::Join);
  TEST_ASSERT(retirement.phase() == TaskRetirementPhase::Retiring);
  TEST_ASSERT(retirement.claim(true) == TaskRetirementClaim::Wait);
  retirement.publish();
  TEST_ASSERT(retirement.phase() == TaskRetirementPhase::Retired);
  TEST_ASSERT(retirement.retired());
  TEST_ASSERT(retirement.claim(true) == TaskRetirementClaim::Retired);
  retirement.reset();
  TEST_ASSERT(retirement.phase() == TaskRetirementPhase::Live);
  TEST_ASSERT(retirement.claim(false) == TaskRetirementClaim::Retired);
  TEST_ASSERT(retirement.retired());

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

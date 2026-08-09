#include "test/assert.hpp"

#include "src/compute/job/cpu/model.hpp"
#include "src/runtime/compute/local.hpp"
#include "src/runtime/compute/operation.hpp"
#include "src/runtime/compute/state.hpp"
#include "src/runtime/compute/terminal.hpp"

#include <node/runtime/compute/access.hpp>
#include <rund/compute.hpp>

#include <array>
#include <atomic>
#include <barrier>
#include <cstdint>
#include <memory>
#include <mutex>
#include <span>
#include <string_view>
#include <thread>
#include <type_traits>

namespace {

using ComputeTelemetryEmit =
    void (*)(void *, const rund::compute::Status &,
             const rund::compute::telemetry::Profile &) noexcept;

static_assert(std::is_same_v<rund::node::runtime_detail::ComputeHostState::Emit,
                             ComputeTelemetryEmit>);

using rund::compute::detail::CpuJobProgress;
using rund::compute::detail::CpuJobProgressDisposition;
using rund::compute::detail::CpuPassDisposition;
using rund::compute::detail::CpuPassResult;
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

static_assert(!std::is_default_constructible_v<CpuPassResult>);
static_assert(!std::is_aggregate_v<CpuPassResult>);
static_assert(std::is_trivially_copyable_v<CpuPassResult>);

constexpr CpuPassResult FailedCpuPass = CpuPassResult::failed(
    rund::compute::Status::fail(rund::compute::Reason::Cancelled));
constexpr CpuPassResult NextCpuPass = CpuPassResult::next();
constexpr CpuPassResult RepeatCpuPass = CpuPassResult::repeat();
static_assert(FailedCpuPass.disposition() == CpuPassDisposition::Failed);
static_assert(FailedCpuPass.status().reason() ==
              rund::compute::Reason::Cancelled);
static_assert(NextCpuPass.disposition() == CpuPassDisposition::Next);
static_assert(NextCpuPass.status());
static_assert(RepeatCpuPass.disposition() == CpuPassDisposition::Repeat);
static_assert(RepeatCpuPass.status());
static_assert(
    CpuPassResult::failed(rund::compute::Status::success()).status().reason() ==
    rund::compute::Reason::CpuStepInvalid);

static_assert(!std::is_default_constructible_v<CpuJobProgress>);
static_assert(!std::is_aggregate_v<CpuJobProgress>);
static_assert(!std::is_copy_constructible_v<CpuJobProgress>);
static_assert(!std::is_copy_assignable_v<CpuJobProgress>);
static_assert(std::is_nothrow_move_constructible_v<CpuJobProgress>);
static_assert(!std::is_move_assignable_v<CpuJobProgress>);
static_assert(
    std::is_nothrow_move_constructible_v<rund::compute::detail::RunState>);
static_assert(
    std::is_nothrow_move_assignable_v<rund::compute::detail::RunState>);

std::atomic<std::uint32_t> configured_abort_cancels{0u};
std::atomic<std::uint32_t> configured_abort_retires{0u};

struct TerminalProbe final {
  rund::compute::Status status =
      rund::compute::Status::fail(rund::compute::Reason::CompletionInvalid);
  rund::compute::Stats stats{};
  rund::compute::MemoryStats memory{};
  std::uint32_t calls{};
};

void CaptureTerminal(
    void *const raw, const rund::compute::Status &status,
    const rund::compute::telemetry::Profile &profile) noexcept {
  auto &probe = *static_cast<TerminalProbe *>(raw);
  probe.status = status;
  probe.stats = profile.execution();
  probe.memory = profile.memory();
  ++probe.calls;
}

void CountConfiguredAbortCancel(
    const std::shared_ptr<rund::node::runtime_detail::ComputeHostState>
        &) noexcept {
  configured_abort_cancels.fetch_add(1u, std::memory_order_relaxed);
}

void CountConfiguredAbortRetire(
    const std::shared_ptr<rund::node::runtime_detail::ComputeHostState>
        &) noexcept {
  configured_abort_retires.fetch_add(1u, std::memory_order_relaxed);
}

} // namespace

int RunRuntimeComputeTerminalContract() {
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
  using rund::node::runtime_detail::ComputeHostAdmission;
  using rund::node::runtime_detail::ComputeHostCloseClaim;
  using rund::node::runtime_detail::ComputeHostLifecycle;
  using rund::node::runtime_detail::ComputeHostPhase;

  CpuJobProgress failed_cpu_job = CpuJobProgress::failed(
      rund::compute::Status::fail(rund::compute::Reason::Cancelled));
  TEST_ASSERT(failed_cpu_job.disposition() ==
              CpuJobProgressDisposition::Failed);
  TEST_ASSERT(failed_cpu_job.status().reason() ==
              rund::compute::Reason::Cancelled);
  CpuJobProgress normalized_cpu_job =
      CpuJobProgress::failed(rund::compute::Status::success());
  TEST_ASSERT(normalized_cpu_job.disposition() ==
              CpuJobProgressDisposition::Failed);
  TEST_ASSERT(normalized_cpu_job.status().reason() ==
              rund::compute::Reason::RunInvalid);
  CpuJobProgress pending_cpu_job = CpuJobProgress::pending();
  TEST_ASSERT(pending_cpu_job.disposition() ==
              CpuJobProgressDisposition::Pending);
  TEST_ASSERT(pending_cpu_job.status());

  auto run_owner = std::make_shared<rund::compute::detail::ProgramState>();
  rund::compute::detail::RunState run_payload{};
  run_payload.program = run_owner;
  run_payload.semantic_failure_count = 37u;
  CpuJobProgress complete_cpu_job =
      CpuJobProgress::complete(std::move(run_payload));
  TEST_ASSERT(run_payload.program == nullptr);
  TEST_ASSERT(complete_cpu_job.disposition() ==
              CpuJobProgressDisposition::Complete);
  TEST_ASSERT(complete_cpu_job.status());
  CpuJobProgress moved_cpu_job = std::move(complete_cpu_job);
  TEST_ASSERT(complete_cpu_job.disposition() ==
              CpuJobProgressDisposition::Failed);
  TEST_ASSERT(complete_cpu_job.status().reason() ==
              rund::compute::Reason::RunInvalid);
  TEST_ASSERT(moved_cpu_job.disposition() ==
              CpuJobProgressDisposition::Complete);
  rund::compute::detail::RunState taken_run =
      std::move(moved_cpu_job).take_run();
  TEST_ASSERT(taken_run.program == run_owner);
  TEST_ASSERT(taken_run.semantic_failure_count == 37u);
  TEST_ASSERT(moved_cpu_job.disposition() == CpuJobProgressDisposition::Failed);
  TEST_ASSERT(moved_cpu_job.status().reason() ==
              rund::compute::Reason::RunInvalid);

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
      configured_abort, CountConfiguredAbortCancel, CountConfiguredAbortRetire);
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
  const auto cancelled_observation =
      FinishFailure(task, rund::compute::Status::fail(
                              rund::compute::Reason::PrimitiveBackendFailed));
  const rund::compute::Status cancelled = cancelled_observation.status();
  TEST_ASSERT(cancelled_observation.profile() == nullptr);
  TEST_ASSERT(!cancelled);
  TEST_ASSERT(cancelled.error() == std::string_view{"compute_cancelled"});
  const auto output = job->read();
  TEST_ASSERT(!output);
  TEST_ASSERT(output.error() == std::string_view{"compute_cancelled"});

  auto epoch_job = program->resident(input);
  TEST_ASSERT(epoch_job);
  const auto epoch_job_state =
      rund::compute::detail::JobAccess::state(*epoch_job);
  TerminalProbe job_probe{};
  rund::node::runtime_detail::ComputeHostState job_host{};
  job_host.emit_context = &job_probe;
  job_host.emit = CaptureTerminal;
  TaskState job_task{};
  job_task.host = &job_host;
  job_task.operation = rund::node::compute_detail::make_job(epoch_job_state);
  rund::compute::detail::RunState first_job_run{};
  first_job_run.program = epoch_job_state->program;
  first_job_run.stats = {.backend = rund::compute::Backend::Cpu,
                         .dispatches = 11u,
                         .graph_hash = 101u};
  job_task.job_result.emplace(
      rund::compute::Result<rund::compute::detail::RunState>::success(
          std::move(first_job_run)));
  TEST_ASSERT(rund::compute::detail::queue_job(epoch_job_state));
  std::barrier job_barrier{2};
  std::atomic_bool job_reused{false};
  std::thread job_resubmit{[&] {
    job_barrier.arrive_and_wait();
    rund::compute::detail::RunState second_job_run{};
    second_job_run.program = epoch_job_state->program;
    second_job_run.stats = {.backend = rund::compute::Backend::Cpu,
                            .dispatches = 22u,
                            .graph_hash = 202u};
    job_reused.store(
        rund::compute::detail::queue_job(epoch_job_state) &&
            rund::compute::detail::finish_job(
                epoch_job_state,
                rund::compute::Result<rund::compute::detail::RunState>::success(
                    std::move(second_job_run))),
        std::memory_order_release);
  }};
  auto first_job = rund::node::compute_detail::FinishCpu(job_task);
  job_barrier.arrive_and_wait();
  job_resubmit.join();
  rund::node::Complete(&job_task, std::move(first_job));
  TEST_ASSERT(job_reused.load(std::memory_order_acquire));
  TEST_ASSERT(job_probe.calls == 1u && job_probe.status &&
              job_probe.stats.dispatches == 11u &&
              job_probe.stats.graph_hash == 101u &&
              job_probe.memory.scope == rund::compute::MemoryScope::Job);
  TEST_ASSERT(
      job_task.stats.dispatches == 11u && job_task.stats.graph_hash == 101u &&
      rund::compute::detail::job_stats(epoch_job_state).dispatches == 22u);

  auto pipeline_device = rund::compute::open(rund::compute::Target::cpu(1u));
  TEST_ASSERT(pipeline_device);
  auto pipeline_program =
      rund::compute::on(*pipeline_device)
          .map<std::int32_t>("terminal-pipeline-epoch", input.size(),
                             [](auto value) { return value + 1; })
          .compile();
  auto pipeline_input =
      pipeline_device->upload(std::span<const std::int32_t>{input});
  auto pipeline_output = pipeline_device->buffer<std::int32_t>(input.size());
  auto epoch_pipeline =
      pipeline_program && pipeline_input && pipeline_output
          ? rund::compute::pipeline(*pipeline_device)
                .then(*pipeline_program, rund::compute::read(*pipeline_input),
                      rund::compute::write(*pipeline_output))
                .prepare()
          : rund::compute::Result<rund::compute::Pipeline>::fail(
                rund::compute::Reason::PipelineInvalid);
  TEST_ASSERT(epoch_pipeline);
  const auto &epoch_pipeline_state =
      rund::compute::detail::PipelineStateAccess::state(*epoch_pipeline);
  TerminalProbe pipeline_probe{};
  rund::node::runtime_detail::ComputeHostState pipeline_host{};
  pipeline_host.emit_context = &pipeline_probe;
  pipeline_host.emit = CaptureTerminal;
  TaskState pipeline_task{};
  pipeline_task.host = &pipeline_host;
  pipeline_task.operation =
      rund::node::compute_detail::make_pipeline(epoch_pipeline_state);
  TEST_ASSERT(rund::compute::detail::queue_pipeline(epoch_pipeline_state));
  {
    std::lock_guard lock{epoch_pipeline_state->gate};
    epoch_pipeline_state->stats.dispatches = 33u;
    epoch_pipeline_state->stats.graph_hash = 303u;
  }
  std::barrier pipeline_barrier{2};
  std::atomic_bool pipeline_reused{false};
  std::thread pipeline_resubmit{[&] {
    pipeline_barrier.arrive_and_wait();
    const bool queued = static_cast<bool>(
        rund::compute::detail::queue_pipeline(epoch_pipeline_state));
    if (queued) {
      std::lock_guard lock{epoch_pipeline_state->gate};
      epoch_pipeline_state->stats.dispatches = 44u;
      epoch_pipeline_state->stats.graph_hash = 404u;
    }
    const bool finished =
        queued && !rund::compute::detail::fail_pipeline(
                      epoch_pipeline_state,
                      rund::compute::Status::fail(
                          rund::compute::Reason::PrimitiveBackendFailed));
    pipeline_reused.store(finished, std::memory_order_release);
  }};
  auto first_pipeline = rund::node::compute_detail::FinishFailure(
      pipeline_task, rund::compute::Status::fail(
                         rund::compute::Reason::PrimitiveBackendFailed));
  pipeline_barrier.arrive_and_wait();
  pipeline_resubmit.join();
  rund::node::Complete(&pipeline_task, std::move(first_pipeline));
  TEST_ASSERT(pipeline_reused.load(std::memory_order_acquire));
  TEST_ASSERT(pipeline_probe.calls == 1u && !pipeline_probe.status &&
              pipeline_probe.stats.dispatches == 33u &&
              pipeline_probe.stats.graph_hash == 303u &&
              pipeline_probe.memory.scope ==
                  rund::compute::MemoryScope::Pipeline);
  TEST_ASSERT(
      pipeline_task.stats.dispatches == 33u &&
      pipeline_task.stats.graph_hash == 303u &&
      rund::compute::detail::pipeline_stats(epoch_pipeline_state).dispatches ==
          44u);
  return 0;
}

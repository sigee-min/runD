#include "../local.hpp"

#if !defined(RUND_NODE_TEST_BACKEND_CPU) &&                                    \
    !defined(RUND_NODE_TEST_BACKEND_VULKAN)

namespace rund_node_test_pipeline {

int CheckSingleExecutionAdapter(
    const std::shared_ptr<rund::compute::detail::PipelineState> &state,
    execution::Owner &owner, const execution::SealResult &sealed) {
  using namespace rund::compute;
  execution::Control control{};
  execution::NativeEvidence produced{};
  NativeWait wait{};
  detail::PipelineExecutionSubmission submission{};
  control.evidence = &produced;
  control.completion = CompleteNative;
  control.user = &wait;

  const auto generation = [&state]() noexcept {
    std::lock_guard state_lock{state->gate};
    if (state->publication == nullptr) {
      return std::uint64_t{0u};
    }
    std::lock_guard publication_lock{state->publication->gate};
    return state->publication->generation;
  };
  const detail::DeviceOps *const original_ops = state->device->ops;
  const std::uint64_t guarded_generation = generation();

  // Raw backend acceptance is not a Pipeline terminal. A tampered native
  // control must finish through the shared Pipeline SOT and suppress the
  // public generation before NativeEvidence reaches the run owner.
  detail::DeviceOps injected_ops = *original_ops;
  injected_ops.residency.submit_residency_pipeline = TamperPipelineExecution;
  state->device->ops = &injected_ops;
  struct CompletionCount final {
    NativeWait *wait;
    std::atomic<unsigned> count{0u};
  } callbacks{.wait = &wait};
  control.user = &callbacks;
  control.completion = [](void *raw,
                          execution::NativeEvidence &&value) noexcept {
    auto &observed = *static_cast<CompletionCount *>(raw);
    observed.count.fetch_add(1u, std::memory_order_relaxed);
    CompleteNative(observed.wait, std::move(value));
  };
  control.token = 1u;
  control.generation = 1u;
  wait.done.store(false, std::memory_order_relaxed);
  wait.evidence = {};
  produced = {};
  const Status tampered = detail::submit_pipeline_execution(
      state, owner, sealed.plan, control, submission);
  const bool completed_inside_submit =
      wait.done.load(std::memory_order_acquire);
  if (tampered) {
    wait.done.wait(false, std::memory_order_acquire);
  }
  state->device->ops = original_ops;
  const bool tamper_rejected =
      tampered && completed_inside_submit &&
      callbacks.count.load(std::memory_order_relaxed) == 1u &&
      wait.evidence.status.reason() == Reason::CompletionInvalid &&
      wait.evidence.native_submissions == 1u &&
      wait.evidence.native_completions == 1u &&
      generation() == guarded_generation &&
      state->phase == detail::PipelinePhase::Ready &&
      !control.active.load(std::memory_order_acquire);
  submission.reset();
  if (!tamper_rejected) {
    return 8;
  }

  control.user = &wait;
  control.completion = CompleteNative;

  // A synchronous submit rejection emits no callback, does not advance the
  // Pipeline publication, and returns the fixed submission owner immediately.
  injected_ops = *original_ops;
  injected_ops.residency.submit_residency_pipeline = RejectPipelineExecution;
  state->device->ops = &injected_ops;
  control.token = 2u;
  control.generation = 2u;
  wait.done.store(false, std::memory_order_relaxed);
  const Status rejected_submit = detail::submit_pipeline_execution(
      state, owner, sealed.plan, control, submission);
  state->device->ops = original_ops;
  if (rejected_submit.reason() != Reason::BackendFailed ||
      wait.done.load(std::memory_order_acquire) || submission.active() ||
      control.active.load(std::memory_order_acquire) ||
      generation() != guarded_generation ||
      state->phase != detail::PipelinePhase::Ready) {
    return 8;
  }

  bool clean = true;
  Reason failure = Reason::Ok;
  std::size_t failed_sample = 0u;
  const auto submit = [&](const std::size_t sample) noexcept {
    failed_sample = sample;
    wait.done.store(false, std::memory_order_relaxed);
    wait.evidence = {};
    produced = {};
    control.token = sample;
    control.generation = sample;
    const Status submitted = detail::submit_pipeline_execution(
        state, owner, sealed.plan, control, submission);
    if (!submitted) {
      failure = submitted.reason();
      return false;
    }
    wait.done.wait(false, std::memory_order_acquire);
    const execution::NativeEvidence &evidence = wait.evidence;
    const bool valid =
        evidence.status &&
        evidence.terminal == execution::TerminalKind::Known &&
        evidence.plan_identity == sealed.plan.identity() &&
        evidence.token == control.token &&
        evidence.generation == control.generation &&
        evidence.epoch_count == 1u && evidence.native_submissions == 1u &&
        evidence.native_dispatches == 1u && evidence.native_completions == 1u &&
        evidence.native_inflight_peak == 1u && evidence.failure_count == 0u &&
        !control.active.load(std::memory_order_acquire);
    if (!valid) {
      failure = evidence.status.reason();
    }
    submission.reset();
    return valid;
  };

  // The retained native route may initialize backend-owned objects on its
  // first real submission. That is cold preparation, not a warm allocation.
  clean = submit(1u);
  node_compute_allocation::Start();
  constexpr std::size_t warm_runs = 60u;
  for (std::size_t sample = 0u; clean && sample < warm_runs; ++sample) {
    clean = submit(sample + 2u);
  }
  node_compute_allocation::Stop();
  if (!clean || node_compute_allocation::Count() != 0u) {
    std::fprintf(
        stderr,
        "execution adapter sample=%zu failure=%u allocation=%llu status=%u "
        "terminal=%u plan=%llu/%llu token=%llu/%llu generation=%llu/%llu "
        "submit=%llu dispatch=%llu complete=%llu peak=%llu failures=%zu "
        "active=%u\n",
        failed_sample, static_cast<unsigned>(failure),
        static_cast<unsigned long long>(node_compute_allocation::Count()),
        static_cast<unsigned>(wait.evidence.status.reason()),
        static_cast<unsigned>(wait.evidence.terminal),
        static_cast<unsigned long long>(wait.evidence.plan_identity),
        static_cast<unsigned long long>(sealed.plan.identity()),
        static_cast<unsigned long long>(wait.evidence.token),
        static_cast<unsigned long long>(control.token),
        static_cast<unsigned long long>(wait.evidence.generation),
        static_cast<unsigned long long>(control.generation),
        static_cast<unsigned long long>(wait.evidence.native_submissions),
        static_cast<unsigned long long>(wait.evidence.native_dispatches),
        static_cast<unsigned long long>(wait.evidence.native_completions),
        static_cast<unsigned long long>(wait.evidence.native_inflight_peak),
        wait.evidence.failure_count,
        static_cast<unsigned>(control.active.load(std::memory_order_acquire)));
    return 8;
  }
  return 0;
}

} // namespace rund_node_test_pipeline

#endif

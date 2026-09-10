#include "local.hpp"

#if !defined(RUND_NODE_TEST_BACKEND_CPU) &&                                    \
    !defined(RUND_NODE_TEST_BACKEND_VULKAN)

namespace rund_node_test_pipeline {

[[nodiscard]] int CheckExecutionAdapter() {
  using namespace rund::compute;
  auto opened = open(Target::metal());
  if (!opened) {
    return opened.reason() == Reason::AdapterUnavailable ? 0 : 1;
  }
  Device device = std::move(opened).value();
  constexpr std::size_t elements = 1u;
  auto program = on(device)
                     .map<std::int32_t>("metal-execution-adapter", elements,
                                        [](auto value) { return value + 1; })
                     .compile();
  auto input_backing =
      std::make_shared<AdmissionBacking>(elements * sizeof(std::int32_t));
  auto output_backing =
      std::make_shared<AdmissionBacking>(elements * sizeof(std::int32_t));
  auto input = detail::make_virtual_buffer(
      elements, sizeof(std::int32_t), detail::Type::I32, {}, input_backing);
  auto output = detail::make_virtual_buffer(
      elements, sizeof(std::int32_t), detail::Type::I32, {}, output_backing);
  if (!program || !input || !output) {
    return 2;
  }
  auto prepared = detail::prepare_virtual_pipeline(
      detail::ProgramAccess::state(*program), std::move(input).value(),
      std::move(output).value(), ResidencyConfig{});
  if (!prepared) {
    return 3;
  }
  const std::shared_ptr<detail::PipelineState> state =
      prepared.value()->pipeline;
  if (state == nullptr || state->device == nullptr ||
      state->device->ops == nullptr ||
      state->device->ops->residency.prepare_residency_selection == nullptr ||
      state->device->ops->residency.prepare_residency_execution == nullptr ||
      state->residency_pool == nullptr) {
    return 4;
  }
  bool supported = false;
  if (!rund::node::accel::detail::QueryPreparedKernelPipelineResidency(
          state->prepared, supported)) {
    return 5;
  }
  if (!supported) {
    return 0;
  }
  const execution::SealResult sealed = ExecutionPlan(*state->residency_pool);
  if (!sealed || sealed.plan.epoch_count() != 1u) {
    return 7;
  }

  const std::array<const rund::node::accel::detail::PreparedKernelPipeline *,
                   1u>
      pipelines{&state->prepared};
  execution::Owner owner{};
  const Status bound =
      state->device->ops->residency.prepare_residency_execution(
          *state->device, pipelines, owner);
  std::uint64_t retained = 1u;
  if (!bound || !owner || !owner.retained_bytes(retained) || retained != 0u) {
    return 8;
  }

  if (const int single = CheckSingleExecutionAdapter(state, owner, sealed);
      single != 0) {
    return single;
  }

  // The bounded adapter accepts one fixed Q=4 handoff, queues four actual
  // Metal batches behind exact ready values, and emits four internal Releases
  // plus one Final. e0/e2 and e1/e3 deliberately reuse prepared owners; each
  // later generation is seeded only at its ready signal after the prior bank
  // Release. A wrong asymmetric e2 generation is rejected before readiness.
  const std::shared_ptr<detail::PipelineState> alternate =
      prepared.value()->alternate_pipeline;
  detail::AccelDeviceState *const native = detail::accel_device(*state->device);
  bool alternate_supported = false;
  if (alternate == nullptr || native == nullptr || !alternate->prepared.ok ||
      !rund::node::accel::detail::QueryPreparedKernelPipelineResidency(
          alternate->prepared, alternate_supported)) {
    return 9;
  }
  if (!alternate_supported) {
    return 0;
  }
  auto claim_input = detail::make_virtual_buffer(
      elements, sizeof(std::int32_t), detail::Type::I32, {},
      std::make_shared<AdmissionBacking>(elements * sizeof(std::int32_t)));
  auto claim_output = detail::make_virtual_buffer(
      elements, sizeof(std::int32_t), detail::Type::I32, {},
      std::make_shared<AdmissionBacking>(elements * sizeof(std::int32_t)));
  auto claim_prepared =
      claim_input && claim_output
          ? detail::prepare_virtual_pipeline(
                detail::ProgramAccess::state(*program),
                std::move(claim_input).value(), std::move(claim_output).value(),
                ResidencyConfig{})
          : Result<std::shared_ptr<detail::VirtualPipelineState>>::fail(
                Reason::PipelineInvalid);
  const std::shared_ptr<detail::PipelineState> claim_primary =
      claim_prepared ? claim_prepared.value()->pipeline : nullptr;
  const std::shared_ptr<detail::PipelineState> claim_alternate =
      claim_prepared ? claim_prepared.value()->alternate_pipeline : nullptr;
  if (claim_primary == nullptr || claim_alternate == nullptr ||
      !claim_primary->prepared.ok || !claim_alternate->prepared.ok) {
    return 9;
  }
  const std::array<rund::node::accel::detail::PreparedKernelPipeline, 4u>
      stream_owners{state->prepared, alternate->prepared,
                    claim_primary->prepared, claim_alternate->prepared};
  WindowWait window{};
  window.context = native->context;
  window.pipelines = {state->prepared, alternate->prepared};
  window.plan_identity = sealed.plan.identity();
  bool window_ready = false;
  const rund::AccelCheck queried =
      rund::node::accel::detail::PreparedKernelPipelineWindowReady(
          window.context, window.pipelines, window_ready);
  if (!queried.ok || !window_ready) {
    return 9;
  }
  rund::node::accel::detail::PreparedResidencyWindowControl window_control{};
  const auto submit_window = [&](const std::size_t sample,
                                 const std::size_t suppressed =
                                     std::numeric_limits<std::size_t>::max(),
                                 const bool terminal_loss = false,
                                 rund::node::accel::detail::
                                     PreparedResidencyStreamControl *const
                                         stream = nullptr) noexcept {
    window.done.store(false, std::memory_order_relaxed);
    window.valid.store(true, std::memory_order_relaxed);
    window.wrong_rejected.store(false, std::memory_order_relaxed);
    window.release_count.store(0u, std::memory_order_relaxed);
    window.final = {};
    window.token = 1000u + sample;
    window.generation = 2000u + sample;
    window.control_base = static_cast<std::uint32_t>(3000u + sample * 4u);
    window.suppressed = suppressed;
    window.terminal_loss = terminal_loss;
    window.aborted = terminal_loss;
    rund::node::accel::detail::PreparedResidencyWindowRequest request{
        .plan_identity = window.plan_identity,
        .token = window.token,
        .generation = window.generation,
        .batch_count = 4u,
        .release = CompleteWindowRelease,
        .final = CompleteWindowFinal,
        .user = &window,
    };
    for (std::size_t index = 0u; index < request.batch_count; ++index) {
      request.batches[index].pipeline = window.pipelines[index % 2u];
      request.batches[index].locals[0u] = 0u;
      request.batches[index].local_count = 1u;
      request.batches[index].epoch = index;
      request.batches[index].control_generation =
          window.control_base + static_cast<std::uint32_t>(index);
      request.batches[index].bank = static_cast<std::uint8_t>(index % 2u);
    }
    if (terminal_loss &&
        !rund::node::accel::detail::InjectNativeResidencyTerminalLossOnce(
            native->pick)) {
      return false;
    }
    const rund::AccelCheck submitted =
        stream == nullptr
            ? rund::node::accel::detail::SubmitPreparedKernelPipelineWindow(
                  window.context, request, window_control)
            : rund::node::accel::detail::
                  SubmitPreparedKernelPipelineStreamWindow(
                      window.context, request, window_control, *stream);
    if (!submitted.ok) {
      return false;
    }
    // Release and raw submission serialize on the exact prepared submission
    // mutexes. Once the raw window is active, the stream sentinel must remain
    // attached until its Final clears every submission.window pointer.
    if (stream != nullptr &&
        rund::node::accel::detail::ReleasePreparedKernelPipelineStream(
            *stream, window.plan_identity, window.token, window.generation,
            false)
            .ok) {
      return false;
    }
    bool claimed_ready = true;
    const rund::AccelCheck claimed =
        rund::node::accel::detail::PreparedKernelPipelineWindowReady(
            window.context, window.pipelines, claimed_ready);
    if (!claimed.ok || claimed_ready) {
      return false;
    }
    // Signal bank 1 first. Queue execution is blocked on e0, making the
    // adapter-observed native ready/in-flight peak exactly two when e0 follows.
    for (const std::size_t index : {1u, 0u}) {
      if (terminal_loss) {
        break;
      }
      const rund::AccelCheck signaled =
          rund::node::accel::detail::SignalPreparedKernelPipelineWindow(
              window.context, window.pipelines[index],
              rund::node::accel::detail::BackendResidencyWindowSignal{
                  .plan_identity = window.plan_identity,
                  .token = window.token,
                  .generation = window.generation,
                  .epoch = index,
                  .control_generation =
                      window.control_base + static_cast<std::uint32_t>(index),
                  .bank = static_cast<std::uint8_t>(index),
              });
      if (!signaled.ok) {
        return false;
      }
    }
    if (terminal_loss) {
      for (std::size_t attempt = 0u; attempt < 2u; ++attempt) {
        const rund::AccelCheck rejected =
            rund::node::accel::detail::SignalPreparedKernelPipelineWindow(
                window.context, window.pipelines[0u],
                rund::node::accel::detail::BackendResidencyWindowSignal{
                    .plan_identity = window.plan_identity,
                    .token = window.token,
                    .generation = window.generation,
                    .epoch = 0u,
                    .control_generation = window.control_base + 1u,
                    .bank = 0u,
                });
        if (rejected.ok) {
          return false;
        }
      }
      window.wrong_rejected.store(true, std::memory_order_release);
      const rund::AccelCheck aborted =
          rund::node::accel::detail::AbortPreparedKernelPipelineWindow(
              window.context, window_control,
              rund::node::accel::detail::BackendResidencyWindowAbort{
                  .failure = {false, "compute_backend_failed"},
                  .plan_identity = window.plan_identity,
                  .token = window.token,
                  .generation = window.generation,
              });
      if (!aborted.ok) {
        return false;
      }
    }
    window.done.wait(false, std::memory_order_acquire);
    const auto &final = window.final;
    const bool failed = suppressed < 4u || terminal_loss;
    bool valid =
        window.valid.load(std::memory_order_acquire) &&
        window.wrong_rejected.load(std::memory_order_acquire) &&
        window.release_count.load(std::memory_order_acquire) == 4u &&
        final.check.ok == !failed &&
        final.terminal ==
            (terminal_loss
                 ? rund::node::accel::detail::NativeTerminal::UnknownMayWrite
                 : rund::node::accel::detail::NativeTerminal::Known) &&
        final.plan_identity == window.plan_identity &&
        final.token == window.token && final.generation == window.generation &&
        final.public_handoffs == 1u && final.native_batches == 4u &&
        final.queue_calls == 4u && final.native_inflight_peak == 2u &&
        final.receipt_count == 4u && final.completed_ns != 0u &&
        !window_control.active && window_control.quarantined == terminal_loss;
    for (std::size_t index = 0u; valid && index < 4u; ++index) {
      const auto &receipt = final.receipts[index];
      const bool row_suppressed = index == suppressed;
      const bool row_lost = terminal_loss;
      valid =
          receipt.check.ok == (!row_suppressed && !row_lost) &&
          receipt.terminal ==
              (row_lost
                   ? rund::node::accel::detail::NativeTerminal::UnknownMayWrite
                   : rund::node::accel::detail::NativeTerminal::Known) &&
          receipt.epoch == index && receipt.backend_sequence == index + 1u &&
          receipt.bank == index % 2u && receipt.dispatched &&
          receipt.completed == !row_lost &&
          receipt.may_write == !row_suppressed;
    }
    return valid;
  };
  bool clean = submit_window(0u);
  node_compute_allocation::Start();
  constexpr std::size_t warm_runs = 60u;
  for (std::size_t sample = 0u; clean && sample < warm_runs; ++sample) {
    clean = submit_window(sample + 1u);
  }
  node_compute_allocation::Stop();
  if (!clean || node_compute_allocation::Count() != 0u) {
    std::fprintf(
        stderr,
        "metal window valid=%u wrong=%u releases=%zu allocation=%llu "
        "handoff=%llu batches=%llu queues=%llu peak=%llu final=%s\n",
        static_cast<unsigned>(window.valid.load(std::memory_order_acquire)),
        static_cast<unsigned>(
            window.wrong_rejected.load(std::memory_order_acquire)),
        window.release_count.load(std::memory_order_acquire),
        static_cast<unsigned long long>(node_compute_allocation::Count()),
        static_cast<unsigned long long>(window.final.public_handoffs),
        static_cast<unsigned long long>(window.final.native_batches),
        static_cast<unsigned long long>(window.final.queue_calls),
        static_cast<unsigned long long>(window.final.native_inflight_peak),
        window.final.check.reason);
    return 9;
  }

  // The source-private recurrent sentinel owns all four parity-reachable
  // prepared states. A peer claim is Busy, an active raw window cannot detach
  // it, and an intermediate Known Final clears only the bounded window claim.
  // The sole stream release then makes every owner immediately reclaimable.
  rund::node::accel::detail::PreparedResidencyStreamControl stream_claim{};
  rund::node::accel::detail::PreparedResidencyStreamControl peer_claim{};
  constexpr std::uint64_t stream_token = 9001u;
  constexpr std::uint64_t stream_generation = 10001u;
  if (!rund::node::accel::detail::ClaimPreparedKernelPipelineStream(
           window.context, stream_owners, window.plan_identity, stream_token,
           stream_generation, stream_claim)
           .ok ||
      rund::node::accel::detail::ClaimPreparedKernelPipelineStream(
          window.context, stream_owners, window.plan_identity,
          stream_token + 1u, stream_generation + 1u, peer_claim)
          .ok) {
    return 9;
  }
  if (!submit_window(stream_token - 1000u,
                     std::numeric_limits<std::size_t>::max(), false,
                     &stream_claim) ||
      !stream_claim.active || stream_claim.quarantined ||
      !rund::node::accel::detail::ReleasePreparedKernelPipelineStream(
           stream_claim, window.plan_identity, window.token, window.generation,
           false)
           .ok ||
      stream_claim.active || stream_claim.quarantined ||
      !rund::node::accel::detail::ClaimPreparedKernelPipelineStream(
           window.context, stream_owners, window.plan_identity,
           stream_token + 2u, stream_generation + 2u, stream_claim)
           .ok ||
      !rund::node::accel::detail::ReleasePreparedKernelPipelineStream(
           stream_claim, window.plan_identity, stream_token + 2u,
           stream_generation + 2u, false)
           .ok) {
    return 9;
  }
  if (!submit_window(warm_runs + 1u, 2u) || window.final.check.ok ||
      !window.final.receipts[0u].check.ok ||
      !window.final.receipts[1u].check.ok ||
      window.final.receipts[2u].check.ok ||
      window.final.receipts[2u].may_write ||
      !window.final.receipts[3u].check.ok ||
      !window.final.receipts[3u].may_write) {
    return 9;
  }

  // The legacy Q=1 whole-execution owner must reject Q>1. A recurrent request
  // may not allocate another owner or fall through to the per-epoch selected
  // path; the separate fixed window seam above is the sole W<=4 authority.
  const execution::SealResult recurrent =
      ExecutionPlan(*state->residency_pool, 2u);
  NativeWait wait{};
  detail::PipelineExecutionSubmission submission{};
  execution::Control rejected{};
  execution::NativeEvidence rejected_evidence{};
  rejected.token = 1u;
  rejected.generation = 1u;
  rejected.evidence = &rejected_evidence;
  rejected.completion = CompleteNative;
  rejected.user = &wait;
  if (!recurrent || recurrent.plan.epoch_count() != 2u ||
      detail::submit_pipeline_execution(state, owner, recurrent.plan, rejected,
                                        submission)
              .reason() != Reason::BackendUnsupported ||
      rejected.active.load(std::memory_order_acquire)) {
    return 10;
  }

  // Two rejected stale readiness attempts are followed by the exact accepted
  // window abort. All unopened gates drain conservatively UnknownMayWrite; the
  // injected missing last event additionally proves the watchdog path reaches
  // the same one Unknown Final after all four actual queue calls. The common
  // control pointer is released while prepared/native owners remain
  // self-quarantined, and no success completion is fabricated.
  rund::node::accel::detail::PreparedResidencyStreamControl lost_stream{};
  constexpr std::uint64_t lost_stream_token = 10001u;
  constexpr std::uint64_t lost_stream_generation = 11001u;
  if (!rund::node::accel::detail::ClaimPreparedKernelPipelineStream(
           window.context, stream_owners, window.plan_identity,
           lost_stream_token, lost_stream_generation, lost_stream)
           .ok ||
      !submit_window(lost_stream_token - 1000u,
                     std::numeric_limits<std::size_t>::max(), true,
                     &lost_stream) ||
      window.final.receipts[3u].terminal !=
          rund::node::accel::detail::NativeTerminal::UnknownMayWrite ||
      window.final.receipts[3u].completed ||
      !window.final.receipts[3u].may_write || !lost_stream.active ||
      !lost_stream.quarantined ||
      !rund::node::accel::detail::ReleasePreparedKernelPipelineStream(
           lost_stream, window.plan_identity, window.token, window.generation,
           true)
           .ok ||
      lost_stream.active || !lost_stream.quarantined ||
      rund::node::accel::detail::ClaimPreparedKernelPipelineStream(
          window.context, stream_owners, window.plan_identity,
          lost_stream_token + 1u, lost_stream_generation + 1u, peer_claim)
          .ok) {
    return 11;
  }
  bool quarantine_ready = true;
  const rund::AccelCheck quarantine_query =
      rund::node::accel::detail::PreparedKernelPipelineWindowReady(
          window.context, window.pipelines, quarantine_ready);
  const rund::AccelCheck stale_signal =
      rund::node::accel::detail::SignalPreparedKernelPipelineWindow(
          window.context, window.pipelines[0u],
          rund::node::accel::detail::BackendResidencyWindowSignal{
              .plan_identity = window.plan_identity,
              .token = window.token,
              .generation = window.generation,
              .epoch = 0u,
              .control_generation = window.control_base,
              .bank = 0u,
          });
  if (!quarantine_query.ok || quarantine_ready || stale_signal.ok ||
      window_control.active || !window_control.quarantined) {
    return 11;
  }
  submission.reset();
  return 0;
}

} // namespace rund_node_test_pipeline

#endif

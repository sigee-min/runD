#include "local.hpp"

#include <cstdio>

namespace rund_node_test_pipeline_residency {
namespace {

namespace detail = window::detail;
namespace execution = window::execution;
namespace residency = window::residency;
using window::frames;
using window::pipeline;
using window::plan;
using window::release;
using window::Reason;
using window::Status;

} // namespace

int CheckWindow() {
  if (const int result = window::CheckStreamWindow(); result != 0) {
    return result;
  }
  if (const int result = window::CheckScheduleWindow(); result != 0) {
    return result;
  }

  const execution::SealResult sealed = plan();
  residency::Authority authority{};
  execution::Window window_instance{};
  if (!sealed || sealed.plan.epoch_count() != execution::WindowCapacity ||
      !frames(authority) || !window_instance.begin(authority, sealed.plan)) {
    return 1;
  }
  detail::PipelineExecutionWindow pipeline_window{};
  if (!pipeline_window.bind(sealed.plan, window_instance.lease())) {
    return 2;
  }

  std::array<residency::ExecutionTicket, execution::WindowCapacity> input{};
  std::array<residency::ExecutionTicket, execution::WindowCapacity> output{};
  execution::WindowEvidence premature{};
  if (!window_instance.issue(0u, execution::Phase::Input, input[0]) ||
      input[0].bindings.size() != 2u || input[0].backing_mask != 1u ||
      input[0].transfer_mask != 1u ||
      !window_instance.terminal(input[0], residency::ExecutionTerminal::Success,
                                Status::success()) ||
      !window_instance.issue(1u, execution::Phase::Input, input[1]) ||
      input[1].bindings.size() != 2u || input[1].backing_mask != 1u ||
      input[1].transfer_mask != 1u ||
      !window_instance.terminal(input[1], residency::ExecutionTerminal::Success,
                                Status::success()) ||
      window_instance.issue(2u, execution::Phase::Input, input[2]) ||
      window_instance.issue(0u, execution::Phase::Output, output[0]) ||
      window_instance.final(premature)) {
    return 3;
  }

  for (std::uint64_t epoch = 0u; epoch < execution::WindowCapacity; ++epoch) {
    const execution::Release ready = release(window_instance.lease(), epoch);
    if (!pipeline_window.terminal(pipeline(ready)) ||
        !window_instance.release(ready) ||
        !window_instance.issue(epoch, execution::Phase::Output, output[epoch]) ||
        output[epoch].bindings.size() != 2u ||
        output[epoch].backing_mask != 1u || output[epoch].transfer_mask != 1u ||
        !window_instance.terminal(output[epoch],
                                  residency::ExecutionTerminal::Success,
                                  Status::success())) {
      return 4;
    }
    if (epoch + execution::BankCapacity < execution::WindowCapacity) {
      const std::uint64_t next = epoch + execution::BankCapacity;
      if (!window_instance.issue(next, execution::Phase::Input, input[next]) ||
          !window_instance.terminal(input[next],
                                    residency::ExecutionTerminal::Success,
                                    Status::success())) {
        return 5;
      }
    }
  }
  const execution::WindowSnapshot ready = window_instance.snapshot();
  execution::WindowEvidence evidence{};
  if (ready.release_prefix != 4u || ready.ready_prefix != 4u || ready.final ||
      !pipeline_window.final(4u, 2u, 100u, evidence) ||
      evidence.public_handoffs != 1u || evidence.native_batches != 4u ||
      evidence.queue_calls != 4u || !window_instance.final(evidence)) {
    return 6;
  }
  const residency::ExecutionClose success = window_instance.close();
  if (!success || !success.success || success.quarantined ||
      success.progress.input_services != 4u ||
      success.progress.native_dispatches != 4u ||
      success.progress.native_completions != 4u ||
      success.progress.output_services != 4u) {
    return 7;
  }

  // Raw bounded evidence uses chunk-local fixed slots but authenticates every
  // receipt with its run-global epoch and backend sequence. A shifted Q9
  // chunk must reject stale local epoch zero and project 4..7 / 5..8 exactly.
  const execution::SealResult shifted_plan = plan(9u);
  const residency::ExecutionLease shifted_lease{
      .failure = residency::AuthorityFailure::None,
      .token = 901u,
      .generation = 902u,
      .plan = shifted_plan.plan.identity(),
      .epochs = shifted_plan.plan.epoch_count(),
  };
  detail::PipelineExecutionWindow shifted{};
  if (!shifted_plan || shifted_plan.plan.epoch_count() != 9u ||
      !shifted.bind(shifted_plan.plan, shifted_lease, 4u, 4u) ||
      shifted.terminal(pipeline(release(shifted_lease, 0u)))) {
    return 8;
  }
  for (std::uint64_t epoch = 4u; epoch < 8u; ++epoch) {
    if (!shifted.terminal(pipeline(release(shifted_lease, epoch)))) {
      return 9;
    }
  }
  execution::WindowEvidence shifted_evidence{};
  if (!shifted.final(1u, 2u, 101u, shifted_evidence) ||
      shifted_evidence.first_epoch != 4u ||
      shifted_evidence.epoch_count != 9u ||
      shifted_evidence.release_count != 4u) {
    return 10;
  }
  for (std::size_t slot = 0u; slot < 4u; ++slot) {
    const std::uint64_t epoch = 4u + slot;
    if (shifted_evidence.releases[slot].epoch != epoch ||
        shifted_evidence.releases[slot].backend_sequence != epoch + 1u) {
      return 11;
    }
  }

  // A second four-page invocation starts with pages 2/3 retained in the two
  // global Host slots. Epoch 0/1 must not evict those future hits merely to
  // stage one-shot misses: coherent backing fill targets Device directly.
  // Epoch 2/3 then reuse the retained Host pages with exact Host-to-Device
  // supply. Output owners were retired and must never expose a previous
  // generation's dirty bytes.
  execution::Window warm{};
  if (!warm.begin(authority, sealed.plan)) {
    return 14;
  }
  for (std::uint64_t epoch = 0u; epoch < execution::WindowCapacity; ++epoch) {
    residency::ExecutionTicket warm_input{};
    residency::ExecutionTicket warm_output{};
    const bool direct = epoch < execution::BankCapacity;
    if (!warm.issue(epoch, execution::Phase::Input, warm_input) ||
        warm_input.bindings.size() != 2u ||
        warm_input.backing_mask != (direct ? 1u : 0u) ||
        warm_input.transfer_mask != (direct ? 0u : 1u) ||
        warm_input.coherent_mask != (direct ? 1u : 0u) ||
        !warm.terminal(warm_input, residency::ExecutionTerminal::Success,
                       Status::success())) {
      std::fprintf(stderr,
                   "window warm epoch=%llu bindings=%zu masks=%u/%u/%u\n",
                   static_cast<unsigned long long>(epoch),
                   warm_input.bindings.size(), warm_input.backing_mask,
                   warm_input.transfer_mask, warm_input.coherent_mask);
      return 15;
    }
    const execution::Release ready = release(warm.lease(), epoch);
    if (!warm.release(ready) ||
        !warm.issue(epoch, execution::Phase::Output, warm_output) ||
        !warm.terminal(warm_output, residency::ExecutionTerminal::Success,
                       Status::success())) {
      return 16;
    }
  }
  execution::WindowEvidence warm_evidence{
      .status = Status::success(),
      .terminal = execution::TerminalKind::Known,
      .plan_identity = warm.lease().plan,
      .token = warm.lease().token,
      .generation = warm.lease().generation,
      .epoch_count = execution::WindowCapacity,
      .public_handoffs = 1u,
      .native_batches = execution::WindowCapacity,
      .queue_calls = 1u,
      .native_inflight_peak = 2u,
      .release_count = execution::WindowCapacity,
      .completed_ns = 101u,
  };
  for (std::uint64_t epoch = 0u; epoch < execution::WindowCapacity; ++epoch) {
    warm_evidence.releases[epoch] = release(warm.lease(), epoch);
  }
  if (!warm.final(warm_evidence) || !warm.close().success) {
    return 17;
  }

  // Fixed-local bootstrap authentication precedes the native handoff. If a
  // later bootstrap row is infeasible, abandoning the issued Input tickets
  // must restore their exact prior cache rows and release the journal with no
  // native Release or Output publication.
  residency::Authority bootstrap_authority{};
  execution::Window bootstrap_window{};
  std::array<residency::ExecutionTicket, 2u> bootstrap_input{};
  if (!frames(bootstrap_authority) ||
      !bootstrap_window.begin(bootstrap_authority, sealed.plan) ||
      !bootstrap_window.issue(0u, execution::Phase::Input,
                              bootstrap_input[0u]) ||
      !bootstrap_window.terminal(bootstrap_input[0u],
                                 residency::ExecutionTerminal::Success,
                                 Status::success()) ||
      !bootstrap_window.issue(1u, execution::Phase::Input,
                              bootstrap_input[1u]) ||
      !bootstrap_window.terminal(bootstrap_input[1u],
                                 residency::ExecutionTerminal::Success,
                                 Status::success()) ||
      !bootstrap_window.abandon()) {
    return 18;
  }
  execution::Window bootstrap_retry{};
  if (!bootstrap_retry.begin(bootstrap_authority, sealed.plan) ||
      !bootstrap_retry.abandon()) {
    return 19;
  }

  // A backend-authenticated Known Final can still fail the Compute/Authority
  // join (for example, malformed aggregate peak evidence). The integrity
  // terminal must publish nothing, empty every reserved row, and make the
  // exact same Pool immediately reusable. UnknownMayWrite is never eligible.
  residency::Authority integrity_authority{};
  execution::Window integrity{};
  detail::PipelineExecutionWindow integrity_pipeline{};
  std::array<residency::ExecutionTicket, execution::WindowCapacity>
      integrity_input{};
  std::array<residency::ExecutionTicket, execution::WindowCapacity>
      integrity_output{};
  if (!frames(integrity_authority) ||
      !integrity.begin(integrity_authority, sealed.plan) ||
      !integrity_pipeline.bind(sealed.plan, integrity.lease())) {
    return 20;
  }
  for (std::uint64_t epoch = 0u; epoch < 2u; ++epoch) {
    if (!integrity.issue(epoch, execution::Phase::Input,
                         integrity_input[epoch]) ||
        !integrity.terminal(integrity_input[epoch],
                            residency::ExecutionTerminal::Success,
                            Status::success())) {
      return 21;
    }
  }
  for (std::uint64_t epoch = 0u; epoch < execution::WindowCapacity; ++epoch) {
    const execution::Release row = release(integrity.lease(), epoch);
    if (!integrity_pipeline.terminal(pipeline(row)) ||
        !integrity.release(row) ||
        !integrity.issue(epoch, execution::Phase::Output,
                         integrity_output[epoch]) ||
        !integrity.terminal(integrity_output[epoch],
                            residency::ExecutionTerminal::Success,
                            Status::success())) {
      return 22;
    }
    const std::uint64_t next = epoch + execution::BankCapacity;
    if (next < execution::WindowCapacity &&
        (!integrity.issue(next, execution::Phase::Input,
                          integrity_input[next]) ||
         !integrity.terminal(integrity_input[next],
                             residency::ExecutionTerminal::Success,
                             Status::success()))) {
      return 23;
    }
  }
  execution::WindowEvidence malformed{};
  if (!integrity_pipeline.final(1u, 2u, 400u, malformed)) {
    return 24;
  }
  malformed.native_inflight_peak = execution::BankCapacity + 1u;
  if (integrity.final(malformed) || !integrity.abandon_final(malformed)) {
    return 25;
  }
  execution::Window integrity_retry{};
  if (!integrity_retry.begin(integrity_authority, sealed.plan) ||
      !integrity_retry.abandon()) {
    return 26;
  }

  residency::Authority rejected_authority{};
  execution::Window rejected{};
  if (!frames(rejected_authority) ||
      !rejected.begin(rejected_authority, sealed.plan)) {
    return 8;
  }
  detail::PipelineExecutionWindow rejected_pipeline{};
  std::array<residency::ExecutionTicket, 2u> rejected_input{};
  residency::ExecutionTicket rejected_output{};
  const execution::Release failed_submit =
      release(rejected.lease(), 0u, Status::fail(Reason::BackendFailed),
              execution::TerminalKind::Known, true, true, true);
  const execution::Release completed_suffix = release(rejected.lease(), 1u);
  const execution::Release stopped2 =
      release(rejected.lease(), 2u, Status::fail(Reason::BackendFailed),
              execution::TerminalKind::Known, false, false, false);
  const execution::Release stopped3 =
      release(rejected.lease(), 3u, Status::fail(Reason::BackendFailed),
              execution::TerminalKind::Known, false, false, false);
  execution::WindowEvidence rejected_evidence{};
  if (!rejected_pipeline.bind(sealed.plan, rejected.lease()) ||
      !rejected.issue(0u, execution::Phase::Input, rejected_input[0]) ||
      !rejected.terminal(rejected_input[0],
                         residency::ExecutionTerminal::Success,
                         Status::success()) ||
      !rejected.issue(1u, execution::Phase::Input, rejected_input[1]) ||
      !rejected.terminal(rejected_input[1],
                         residency::ExecutionTerminal::Success,
                         Status::success()) ||
      !rejected_pipeline.terminal(pipeline(failed_submit)) ||
      !rejected.release(failed_submit) ||
      !rejected_pipeline.terminal(pipeline(completed_suffix)) ||
      !rejected.release(completed_suffix) ||
      !rejected.issue(1u, execution::Phase::Output, rejected_output) ||
      !rejected.terminal(rejected_output, residency::ExecutionTerminal::Failure,
                         Status::fail(Reason::TransferInvalid)) ||
      !rejected_pipeline.terminal(pipeline(stopped2)) ||
      !rejected.release(stopped2) ||
      !rejected_pipeline.terminal(pipeline(stopped3)) ||
      !rejected.release(stopped3) || rejected.snapshot().release_prefix != 0u ||
      rejected.close().failure != residency::AuthorityFailure::Busy ||
      !rejected_pipeline.final(2u, 2u, 200u, rejected_evidence) ||
      !rejected.final(rejected_evidence)) {
    return 9;
  }
  const residency::ExecutionClose known = rejected.close();
  if (!known || known.success || !known.quarantined ||
      known.progress.native_dispatches != 2u ||
      known.progress.native_completions != 2u || known.failure_count != 4u ||
      known.failures[0].may_write != 2u || known.failures[0].terminal != 2u ||
      known.failures[2].may_write != 0u || known.failures[3].may_write != 0u) {
    return 10;
  }

  residency::Authority unknown_authority{};
  execution::Window unknown{};
  if (!frames(unknown_authority) ||
      !unknown.begin(unknown_authority, sealed.plan)) {
    return 11;
  }
  detail::PipelineExecutionWindow unknown_pipeline{};
  residency::ExecutionTicket unknown_input{};
  const execution::Release lost =
      release(unknown.lease(), 0u, Status::fail(Reason::DeviceLost),
              execution::TerminalKind::UnknownMayWrite, true, false, true);
  const execution::Release unknown_stop1 =
      release(unknown.lease(), 1u, Status::fail(Reason::DeviceLost),
              execution::TerminalKind::Known, false, false, false);
  const execution::Release unknown_stop2 =
      release(unknown.lease(), 2u, Status::fail(Reason::DeviceLost),
              execution::TerminalKind::Known, false, false, false);
  const execution::Release unknown_stop3 =
      release(unknown.lease(), 3u, Status::fail(Reason::DeviceLost),
              execution::TerminalKind::Known, false, false, false);
  execution::WindowEvidence unknown_evidence{};
  if (!unknown_pipeline.bind(sealed.plan, unknown.lease()) ||
      !unknown.issue(0u, execution::Phase::Input, unknown_input) ||
      !unknown.terminal(unknown_input, residency::ExecutionTerminal::Success,
                        Status::success()) ||
      !unknown_pipeline.terminal(pipeline(lost)) || !unknown.release(lost) ||
      !unknown.snapshot().unknown || unknown.snapshot().ready_prefix != 0u ||
      unknown.close().failure != residency::AuthorityFailure::Busy ||
      !unknown_pipeline.terminal(pipeline(unknown_stop1)) ||
      !unknown.release(unknown_stop1) ||
      !unknown_pipeline.terminal(pipeline(unknown_stop2)) ||
      !unknown.release(unknown_stop2) ||
      !unknown_pipeline.terminal(pipeline(unknown_stop3)) ||
      !unknown.release(unknown_stop3) ||
      !unknown_pipeline.final(1u, 1u, 300u, unknown_evidence) ||
      !unknown.final(unknown_evidence)) {
    return 12;
  }
  const residency::ExecutionClose uncertain = unknown.close();
  if (!uncertain || uncertain.success || !uncertain.quarantined ||
      uncertain.progress.native_dispatches != 1u ||
      uncertain.progress.native_completions != 0u ||
      uncertain.failure_count != 4u || uncertain.failures[0].may_write != 2u ||
      uncertain.failures[0].terminal != 0u) {
    return 13;
  }
  return 0;
}

} // namespace rund_node_test_pipeline_residency

#include "local.hpp"

namespace rund_node_test_pipeline_residency::window {
namespace {

bool stream_model(const std::uint64_t epochs) {
  const execution::SealResult sealed = plan(epochs);
  residency::Authority authority{};
  execution::Stream stream{};
  if (!sealed || !frames(authority) || !stream.begin(authority, sealed.plan)) {
    return false;
  }
  const residency::ExecutionLease lease = stream.lease();
  std::array<residency::ExecutionTicket, execution::BankCapacity> input{};
  std::array<residency::ExecutionTicket, execution::BankCapacity> output{};
  for (std::uint64_t epoch = 0u; epoch < epochs; ++epoch) {
    const std::size_t bank = epoch % execution::BankCapacity;
    if (!stream.issue(epoch, execution::Phase::Input, input[bank]) ||
        !stream.terminal(input[bank], residency::ExecutionTerminal::Success,
                         Status::success())) {
      return false;
    }
    const execution::Release completed = release(lease, epoch);
    if (!stream.release(completed) ||
        !stream.issue(epoch, execution::Phase::Output, output[bank]) ||
        !stream.terminal(output[bank], residency::ExecutionTerminal::Success,
                         Status::success())) {
      return false;
    }
    const std::uint64_t first = epoch - epoch % execution::WindowCapacity;
    const bool chunk_end =
        epoch + 1u == epochs ||
        epoch % execution::WindowCapacity + 1u == execution::WindowCapacity;
    if (chunk_end) {
      const std::size_t count = static_cast<std::size_t>(epoch - first + 1u);
      if (!stream.chunk(chunk(lease, first, count))) {
        return false;
      }
    }
  }
  execution::StreamEvidence evidence{};
  if (!stream.final(evidence) || evidence.public_handoffs != 1u ||
      evidence.chunk_count != (epochs + execution::WindowCapacity - 1u) /
                                  execution::WindowCapacity ||
      evidence.native_batches != epochs || evidence.released_prefix != epochs ||
      evidence.queue_calls != evidence.chunk_count) {
    return false;
  }
  const residency::ExecutionClose closed = stream.close();
  return closed && closed.success && !closed.quarantined &&
         closed.progress.input_services == epochs &&
         closed.progress.native_dispatches == epochs &&
         closed.progress.native_completions == epochs &&
         closed.progress.output_services == epochs;
}

bool stream_failure(const execution::TerminalKind terminal) {
  constexpr std::uint64_t epochs = 9u;
  const execution::SealResult sealed = plan(epochs);
  residency::Authority authority{};
  execution::Stream stream{};
  if (!sealed || !frames(authority) || !stream.begin(authority, sealed.plan)) {
    return false;
  }
  const residency::ExecutionLease lease = stream.lease();
  for (std::uint64_t epoch = 0u; epoch < 4u; ++epoch) {
    residency::ExecutionTicket input{};
    residency::ExecutionTicket output{};
    const execution::Release completed = release(lease, epoch);
    if (!stream.issue(epoch, execution::Phase::Input, input) ||
        !stream.terminal(input, residency::ExecutionTerminal::Success,
                         Status::success()) ||
        !stream.release(completed) ||
        !stream.issue(epoch, execution::Phase::Output, output) ||
        !stream.terminal(output, residency::ExecutionTerminal::Success,
                         Status::success())) {
      return false;
    }
  }
  if (!stream.chunk(chunk(lease, 0u, 4u))) {
    return false;
  }
  for (std::uint64_t epoch = 4u; epoch < 6u; ++epoch) {
    residency::ExecutionTicket input{};
    if (!stream.issue(epoch, execution::Phase::Input, input) ||
        !stream.terminal(input, residency::ExecutionTerminal::Success,
                         Status::success())) {
      return false;
    }
  }
  const Status failure = Status::fail(Reason::BackendFailed);
  execution::WindowEvidence failed{
      .status = failure,
      .terminal = terminal,
      .plan_identity = lease.plan,
      .token = lease.token,
      .generation = lease.generation,
      .first_epoch = 4u,
      .epoch_count = epochs,
      .public_handoffs = 1u,
      .native_batches = 4u,
      .queue_calls = 1u,
      .native_inflight_peak = 2u,
      .release_count = 4u,
      .completed_ns = 99u,
  };
  for (std::size_t slot = 0u; slot < 4u; ++slot) {
    const std::uint64_t epoch = 4u + slot;
    const bool unknown = terminal == execution::TerminalKind::UnknownMayWrite;
    failed.releases[slot] = release(lease, epoch, failure, terminal, true,
                                    !unknown, unknown ? true : false);
    if (!stream.release(failed.releases[slot])) {
      return false;
    }
  }
  if (!stream.chunk(failed)) {
    return false;
  }
  execution::StreamEvidence evidence{};
  if (!stream.final(evidence) || evidence.status ||
      evidence.terminal != terminal || evidence.chunk_count != 2u ||
      evidence.native_batches != 8u || evidence.released_prefix != 8u) {
    return false;
  }
  const residency::ExecutionClose closed = stream.close();
  return closed && !closed.success &&
         closed.quarantined ==
             (terminal == execution::TerminalKind::UnknownMayWrite);
}

bool stream_boundary_abort() {
  constexpr std::uint64_t epochs = 9u;
  const execution::SealResult sealed = plan(epochs);
  residency::Authority authority{};
  execution::Stream stream{};
  if (!sealed || !frames(authority) || !stream.begin(authority, sealed.plan)) {
    return false;
  }
  const residency::ExecutionLease lease = stream.lease();
  for (std::uint64_t epoch = 0u; epoch < 4u; ++epoch) {
    residency::ExecutionTicket input{};
    residency::ExecutionTicket output{};
    if (!stream.issue(epoch, execution::Phase::Input, input) ||
        !stream.terminal(input, residency::ExecutionTerminal::Success,
                         Status::success()) ||
        !stream.release(release(lease, epoch)) ||
        !stream.issue(epoch, execution::Phase::Output, output) ||
        !stream.terminal(output, residency::ExecutionTerminal::Success,
                         Status::success())) {
      return false;
    }
    // A short non-tail chunk is not an alternate schedule: fixed chunk
    // cardinality is part of the raw/backend accounting contract.
    if (epoch == 1u && stream.chunk(chunk(lease, 0u, 2u))) {
      return false;
    }
  }
  if (!stream.chunk(chunk(lease, 0u, 4u)) ||
      !stream.abort(Status::fail(Reason::PipelineCapacity), 501u)) {
    return false;
  }
  execution::StreamEvidence evidence{};
  if (!stream.final(evidence) || evidence.status ||
      evidence.status.reason() != Reason::PipelineCapacity ||
      evidence.terminal != execution::TerminalKind::Known ||
      evidence.public_handoffs != 1u || evidence.chunk_count != 1u ||
      evidence.native_batches != 4u || evidence.queue_calls != 1u ||
      evidence.released_prefix != 4u) {
    return false;
  }
  // No future epoch was forged and the discarded reservation is immediately
  // reusable by the exact same Authority/Pool.
  execution::Stream retry{};
  return retry.begin(authority, sealed.plan) && retry.abandon();
}

} // namespace

int CheckStreamWindow() {
  const bool stream_model_q5 = stream_model(5u);
  if (!stream_model_q5) {
    return 40;
  }
  const bool stream_model_q9 = stream_model(9u);
  if (!stream_model_q9) {
    return 45;
  }
  const bool stream_model_q100000 = stream_model(100'000u);
  if (!stream_model_q100000) {
    return 46;
  }
  const bool stream_failure_known =
      stream_failure(execution::TerminalKind::Known);
  if (!stream_failure_known) {
    return 47;
  }
  const bool stream_failure_unknown =
      stream_failure(execution::TerminalKind::UnknownMayWrite);
  if (!stream_failure_unknown) {
    return 48;
  }
  const bool stream_boundary_abort_result = stream_boundary_abort();
  if (!stream_boundary_abort_result) {
    return 49;
  }
  return 0;
}

} // namespace rund_node_test_pipeline_residency::window

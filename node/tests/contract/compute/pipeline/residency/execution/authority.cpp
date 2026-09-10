#include "local.hpp"

namespace rund_node_test_pipeline_residency::execution_test {

[[nodiscard]] int CheckExecutionAuthority() {
  const execution::Request request = MakeRequest();
  const execution::SealResult sealed = execution::seal(request);
  residency::Authority authority{};
  if (!sealed || !RegisterOwners(authority)) {
    return 12;
  }
  std::array<std::uint8_t, 1u> resident{};
  const std::array<residency::CacheKey, 1u> cached_keys{InputKey()};
  if (!SeedCachedInput(authority, InputKey()) ||
      !authority.probe(cached_keys, resident, 0u, 3u) || resident[0] != 1u) {
    return 13;
  }
  const residency::ExecutionLease lease =
      authority.executions().begin_execution(sealed.plan);
  if (!lease || lease.plan != sealed.plan.identity() || lease.epochs != 4u ||
      authority
              .begin({}, residency::FrameTier::Device,
                     residency::FrameRole::Input, 6u, 3u)
              .failure != residency::AuthorityFailure::Busy ||
      authority.drain_dirty(0u, 3u).failure !=
          residency::AuthorityFailure::Busy) {
    return 14;
  }
  std::uint64_t sequence = 1u;
  const auto issue = [&](const std::uint64_t epoch,
                         const execution::Phase phase,
                         residency::ExecutionTicket &ticket) {
    execution::Node node{};
    if (!sealed.plan.project(execution::NodeId{.epoch = epoch, .phase = phase},
                             node) ||
        !authority.executions().issue_execution(lease.token, lease.generation, sealed.plan,
                                   node, sequence, ticket)) {
      return false;
    }
    ++sequence;
    return true;
  };
  const auto finish = [&](residency::ExecutionTicket &ticket,
                          const residency::ExecutionTerminal terminal =
                              residency::ExecutionTerminal::Success) {
    if (!authority.executions().terminal_execution(ticket, terminal)) {
      return false;
    }
    return true;
  };
  residency::ExecutionTicket input0{};
  residency::ExecutionTicket input1{};
  residency::ExecutionTicket dispatch0{};
  residency::ExecutionTicket dispatch1{};
  residency::ExecutionTicket input2{};
  residency::ExecutionTicket output0{};
  residency::ExecutionTicket dispatch2{};
  residency::ExecutionTicket input3{};
  residency::ExecutionTicket output1{};
  residency::ExecutionTicket dispatch3{};
  residency::ExecutionTicket output2{};
  residency::ExecutionTicket output3{};
  if (!issue(0u, execution::Phase::Input, input0) || !finish(input0) ||
      authority.executions().terminal_execution(input0,
                                   residency::ExecutionTerminal::Success) ||
      !issue(1u, execution::Phase::Input, input1) || !finish(input1) ||
      !issue(0u, execution::Phase::Dispatch, dispatch0) ||
      !issue(1u, execution::Phase::Dispatch, dispatch1) ||
      issue(2u, execution::Phase::Input, input2) || !finish(dispatch0) ||
      !issue(2u, execution::Phase::Input, input2) || !finish(input2) ||
      issue(2u, execution::Phase::Dispatch, dispatch2) ||
      !issue(0u, execution::Phase::Output, output0) || !finish(output0) ||
      !issue(2u, execution::Phase::Dispatch, dispatch2) || !finish(dispatch1) ||
      !issue(3u, execution::Phase::Input, input3) || !finish(input3) ||
      !issue(1u, execution::Phase::Output, output1) || !finish(output1) ||
      !issue(3u, execution::Phase::Dispatch, dispatch3) || !finish(dispatch2) ||
      !issue(2u, execution::Phase::Output, output2) || !finish(dispatch3) ||
      !issue(3u, execution::Phase::Output, output3) || !finish(output3)) {
    return 15;
  }
  // Output(2) is still outstanding even though the later bank completed.
  // A premature close must leave the journal live until that exact terminal.
  if (authority.executions().close_execution(lease.token, lease.generation, sealed.plan)
              .failure != residency::AuthorityFailure::Busy ||
      !finish(output2)) {
    return 15;
  }
  const residency::ExecutionClose closed =
      authority.executions().close_execution(lease.token, lease.generation, sealed.plan);
  if (!closed || !closed.success || closed.quarantined ||
      closed.failure_count != 0u ||
      closed.progress.input_services != sealed.plan.epoch_count() ||
      closed.progress.native_dispatches != sealed.plan.epoch_count() ||
      closed.progress.native_completions != sealed.plan.epoch_count() ||
      closed.progress.output_services != sealed.plan.epoch_count() ||
      closed.progress.native_inflight_peak != 2u) {
    return 16;
  }
  resident[0] = 1u;
  if (!authority.probe(cached_keys, resident, 0u, 3u) || resident[0] != 0u) {
    return 17;
  }

  const residency::ExecutionLease failed_lease =
      authority.executions().begin_execution(sealed.plan);
  execution::Node failed_node{};
  residency::ExecutionTicket failed_ticket{};
  if (!failed_lease ||
      !sealed.plan.project(
          execution::NodeId{.epoch = 0u, .phase = execution::Phase::Input},
          failed_node) ||
      !authority.executions().issue_execution(failed_lease.token, failed_lease.generation,
                                 sealed.plan, failed_node, 1u, failed_ticket) ||
      !authority.executions().terminal_execution(
          failed_ticket, residency::ExecutionTerminal::UnknownMayWrite) ||
      authority.executions().issue_execution(failed_lease.token, failed_lease.generation,
                                sealed.plan, failed_node, 2u, failed_ticket)) {
    return 18;
  }
  const residency::ExecutionClose failed_close = authority.executions().close_execution(
      failed_lease.token, failed_lease.generation, sealed.plan);
  const residency::ExecutionLease opaque =
      authority.executions().begin_execution(sealed.plan);
  if (!failed_close || failed_close.success || !failed_close.quarantined ||
      failed_close.failure_count != 1u ||
      failed_close.failures[0].epoch != 0u ||
      failed_close.failures[0].phases != 1u || !opaque) {
    return 19;
  }
  const execution::Evidence success_evidence{
      .status = Status::success(),
      .plan_identity = sealed.plan.identity(),
      .token = opaque.token,
      .generation = opaque.generation,
      .epoch_count = sealed.plan.epoch_count(),
      .native_submissions = 1u,
      .progress =
          execution::Progress{
              .input_services = sealed.plan.epoch_count(),
              .native_dispatches = sealed.plan.epoch_count(),
              .native_completions = sealed.plan.epoch_count(),
              .output_services = sealed.plan.epoch_count(),
              .native_inflight_peak = 2u,
          },
  };
  execution::Evidence stale_evidence = success_evidence;
  ++stale_evidence.generation;
  if (authority.executions().close_execution(sealed.plan, stale_evidence).failure !=
      residency::AuthorityFailure::Invalid) {
    return 20;
  }
  const residency::ExecutionClose opaque_success =
      authority.executions().close_execution(sealed.plan, success_evidence);
  if (!opaque_success || !opaque_success.success ||
      opaque_success.progress.output_services != sealed.plan.epoch_count()) {
    return 21;
  }

  constexpr std::uint64_t LargePages = 1'000'003u;
  execution::Request large_request = request;
  large_request.page_count = LargePages;
  large_request.input.cache.page_count = LargePages;
  large_request.input.logical_bytes = LargePages * 128u - 1u;
  large_request.output.cache.page_count = LargePages;
  large_request.output.logical_bytes = LargePages * 80u - 1u;
  large_request.publication.extent.bytes = large_request.output.logical_bytes;
  const execution::SealResult large = execution::seal(large_request);
  const residency::ExecutionLease large_lease =
      authority.executions().begin_execution(large.plan);
  const execution::Evidence large_evidence{
      .status = Status::success(),
      .plan_identity = large.plan.identity(),
      .token = large_lease.token,
      .generation = large_lease.generation,
      .epoch_count = large.plan.epoch_count(),
      .native_submissions = 1u,
      .progress =
          execution::Progress{
              .input_services = large.plan.epoch_count(),
              .native_dispatches = large.plan.epoch_count(),
              .native_completions = large.plan.epoch_count(),
              .output_services = large.plan.epoch_count(),
              .native_inflight_peak = 2u,
          },
  };
  if (!large || large.plan.epoch_count() < 300'000u || !large_lease ||
      !authority.executions().close_execution(large.plan, large_evidence).success) {
    return 22;
  }
  const residency::ExecutionLease unknown =
      authority.executions().begin_execution(sealed.plan);
  execution::Evidence unknown_evidence{
      .status = Status::fail(Reason::DeviceLost),
      .terminal = execution::TerminalKind::UnknownMayWrite,
      .plan_identity = sealed.plan.identity(),
      .token = unknown.token,
      .generation = unknown.generation,
      .epoch_count = sealed.plan.epoch_count(),
      .native_submissions = 1u,
      .progress =
          execution::Progress{
              .input_services = 2u,
              .native_dispatches = 1u,
              .native_completions = 1u,
              .native_inflight_peak = 1u,
          },
      .failure_count = 2u,
  };
  unknown_evidence.failures[0] = execution::FailureEvidence{
      .epoch = 0u, .phases = 1u, .may_write = 1u, .terminal = 0u};
  unknown_evidence.failures[1] = execution::FailureEvidence{
      .epoch = 1u, .phases = 4u, .may_write = 4u, .terminal = 0u};
  const residency::ExecutionClose opaque_unknown =
      authority.executions().close_execution(sealed.plan, unknown_evidence);
  const residency::ExecutionLease retry =
      authority.executions().begin_execution(sealed.plan);
  if (!unknown || !opaque_unknown || opaque_unknown.success ||
      !opaque_unknown.quarantined || opaque_unknown.failure_count != 2u ||
      opaque_unknown.failures[1].phases != 4u || !retry) {
    return 23;
  }
  execution::Evidence retry_evidence = success_evidence;
  retry_evidence.token = retry.token;
  retry_evidence.generation = retry.generation;
  if (!authority.executions().close_execution(sealed.plan, retry_evidence).success) {
    return 24;
  }
  return 0;
}

} // namespace rund_node_test_pipeline_residency::execution_test

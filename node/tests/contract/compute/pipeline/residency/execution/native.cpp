#include "src/compute/device/state.hpp"
#include "src/compute/device/residency/execution/owner.hpp"
#include "local.hpp"

#include <algorithm>
#include <bit>
#include <cstdio>
#include <memory>

namespace rund_node_test_pipeline_residency::execution_test {

[[nodiscard]] int CheckExecutionNative() {
  const execution::Request request = MakeRequest();
  const execution::Request one_request = MakeOneRequest(request);
  const execution::SealResult one = execution::seal(one_request);
  residency::Authority authority{};
  if (!one || !RegisterOwners(authority) ||
      !SeedCachedInput(authority, InputKey())) {
    return 25;
  }
  execution::Run joined{};
  const residency::ExecutionLease joined_lease =
      one ? joined.begin(authority, one.plan) : residency::ExecutionLease{};
  residency::ExecutionTicket joined_input{};
  // SeedCachedInput retained Host page 0. Only pages 1/2 need backing fetch;
  // all Device pages still need supply (two Fetch + two Host + three Device Map).
  if (!one || one.plan.epoch_count() != 1u || !joined_lease ||
      !joined.issue(execution::Phase::Input, joined_input) ||
      joined_input.bindings.size() != 6u ||
      joined_input.transitions.size() != 7u ||
      joined_input.backing_mask != 6u || joined_input.transfer_mask != 7u ||
      joined_input.bindings[0].fetch || !joined_input.bindings[1].fetch ||
      !joined_input.bindings[2].fetch || !joined_input.bindings[3].fetch ||
      !joined_input.bindings[4].fetch || !joined_input.bindings[5].fetch ||
      !joined.terminal(joined_input, residency::ExecutionTerminal::Success,
                       Status::success())) {
    std::fprintf(
        stderr,
        "native input lease=%u bindings=%zu transitions=%zu masks=%u/%u\n",
        static_cast<unsigned>(bool(joined_lease)), joined_input.bindings.size(),
        joined_input.transitions.size(), joined_input.backing_mask,
        joined_input.transfer_mask);
    return 25;
  }
  const execution::NativeEvidence native{
      .status = Status::success(),
      .plan_identity = one.plan.identity(),
      .token = joined_lease.token,
      .generation = joined_lease.generation,
      .epoch_count = 1u,
      .native_submissions = 1u,
      .native_dispatches = 1u,
      .native_completions = 1u,
      .native_inflight_peak = 1u,
      .completed_ns = 500u,
  };
  residency::ExecutionTicket joined_output{};
  if (!joined.native(native) ||
      joined.close(501u).failure != residency::AuthorityFailure::Busy ||
      !joined.issue(execution::Phase::Output, joined_output) ||
      joined_output.bindings.size() != 6u ||
      joined_output.transitions.size() != 6u ||
      joined_output.backing_mask != 7u || joined_output.transfer_mask != 7u ||
      std::count_if(
          joined_output.transitions.begin(), joined_output.transitions.end(),
          [](const residency::CacheTransition transition) {
            return transition.kind == residency::TransitionKind::Writeback;
          }) != 3 ||
      joined.close(502u).failure != residency::AuthorityFailure::Busy ||
      !joined.terminal(joined_output, residency::ExecutionTerminal::Success,
                       Status::success())) {
    return 26;
  }
  const residency::ExecutionClose joined_close = joined.close(503u);
  if (!joined_close || !joined_close.success || joined_close.quarantined ||
      joined_close.progress.input_services != 1u ||
      joined_close.progress.native_dispatches != 1u ||
      joined_close.progress.native_completions != 1u ||
      joined_close.progress.output_services != 1u) {
    return 27;
  }

  execution::Node warm_input_node{};
  if (!one.plan.project(
          execution::NodeId{.epoch = 0u, .phase = execution::Phase::Input},
          warm_input_node)) {
    return 28;
  }
  std::array<residency::CacheKey, 3u> warm_keys{};
  for (std::size_t index = 0u; index < warm_keys.size(); ++index) {
    warm_keys[index] = warm_input_node.input[index].key;
  }
  std::array<std::uint8_t, 3u> warm_host{};
  std::array<std::uint8_t, 3u> warm_device{};
  if (!authority.probe(warm_keys, warm_host, 0u, 3u) ||
      !authority.probe(warm_keys, warm_device, 6u, 3u) ||
      warm_host != std::array<std::uint8_t, 3u>{1u, 1u, 1u} ||
      warm_device != std::array<std::uint8_t, 3u>{1u, 1u, 1u}) {
    return 29;
  }

  // Synchronous native submit rejection has no callback and therefore no
  // NativeEvidence. The explicit no-submit terminal must close the lease with
  // exact zero native progress/may-write and preserve clean Input rows.
  execution::Run no_submit{};
  const residency::ExecutionLease no_submit_lease =
      no_submit.begin(authority, one.plan);
  residency::ExecutionTicket no_submit_input{};
  if (!no_submit_lease ||
      !no_submit.issue(execution::Phase::Input, no_submit_input) ||
      no_submit_input.backing_mask != 0u ||
      no_submit_input.transfer_mask != 0u ||
      !no_submit.terminal(no_submit_input,
                          residency::ExecutionTerminal::Success,
                          Status::success()) ||
      !no_submit.reject(Status::fail(Reason::BackendFailed))) {
    return 30;
  }
  const residency::ExecutionClose no_submit_close = no_submit.close(550u);
  constexpr std::uint8_t dispatch_mask =
      std::uint8_t{1u} << static_cast<std::uint8_t>(execution::Phase::Dispatch);
  if (!no_submit_close || no_submit_close.success ||
      no_submit_close.quarantined ||
      no_submit_close.progress.input_services != 1u ||
      no_submit_close.progress.native_dispatches != 0u ||
      no_submit_close.progress.native_completions != 0u ||
      no_submit_close.progress.output_services != 0u ||
      no_submit_close.failure_count != 1u ||
      no_submit_close.failures[0].phases != dispatch_mask ||
      no_submit_close.failures[0].may_write != 0u ||
      no_submit_close.failures[0].terminal != dispatch_mask ||
      !authority.probe(warm_keys, warm_host, 0u, 3u) ||
      !authority.probe(warm_keys, warm_device, 6u, 3u) ||
      warm_host != std::array<std::uint8_t, 3u>{1u, 1u, 1u} ||
      warm_device != std::array<std::uint8_t, 3u>{1u, 1u, 1u}) {
    return 30;
  }
  std::uint64_t backing_reads = std::popcount(joined_input.backing_mask);
  for (std::uint64_t repeat = 0u; repeat < 60u; ++repeat) {
    execution::Run warm{};
    const residency::ExecutionLease warm_lease =
        warm.begin(authority, one.plan);
    residency::ExecutionTicket warm_input{};
    residency::ExecutionTicket warm_output{};
    execution::NativeEvidence warm_native = native;
    warm_native.token = warm_lease.token;
    warm_native.generation = warm_lease.generation;
    if (!warm_lease || !warm.issue(execution::Phase::Input, warm_input)) {
      return 30;
    }
    backing_reads += std::popcount(warm_input.backing_mask);
    if (warm_input.backing_mask != 0u || warm_input.transfer_mask != 0u ||
        !warm_input.transitions.empty() ||
        std::any_of(warm_input.bindings.begin(), warm_input.bindings.end(),
                    [](const residency::CacheBinding binding) {
                      return binding.fetch;
                    }) ||
        !warm.terminal(warm_input, residency::ExecutionTerminal::Success,
                       Status::success()) ||
        !warm.native(warm_native) ||
        !warm.issue(execution::Phase::Output, warm_output) ||
        !warm.terminal(warm_output, residency::ExecutionTerminal::Success,
                       Status::success()) ||
        !warm.close(600u + repeat).success) {
      return 30;
    }
  }
  if (backing_reads != 2u) {
    return 31;
  }

  execution::Run delayed_failure{};
  const residency::ExecutionLease failure_lease =
      delayed_failure.begin(authority, one.plan);
  residency::ExecutionTicket failure_input{};
  residency::ExecutionTicket failure_output{};
  execution::NativeEvidence failure_native = native;
  failure_native.token = failure_lease.token;
  failure_native.generation = failure_lease.generation;
  if (!failure_lease ||
      !delayed_failure.issue(execution::Phase::Input, failure_input) ||
      !delayed_failure.terminal(failure_input,
                                residency::ExecutionTerminal::Success,
                                Status::success()) ||
      !delayed_failure.native(failure_native) ||
      delayed_failure.close(510u).failure !=
          residency::AuthorityFailure::Busy ||
      !delayed_failure.issue(execution::Phase::Output, failure_output) ||
      !delayed_failure.terminal(failure_output,
                                residency::ExecutionTerminal::Failure,
                                Status::fail(Reason::TransferInvalid))) {
    return 32;
  }
  const residency::ExecutionClose delayed_close = delayed_failure.close(511u);
  if (!delayed_close || delayed_close.success || !delayed_close.quarantined ||
      delayed_close.failure_count != 1u ||
      delayed_close.failures[0].phases != 4u ||
      !authority.probe(warm_keys, warm_host, 0u, 3u) ||
      !authority.probe(warm_keys, warm_device, 6u, 3u) ||
      warm_host != std::array<std::uint8_t, 3u>{1u, 1u, 1u} ||
      warm_device != std::array<std::uint8_t, 3u>{1u, 1u, 1u}) {
    return 33;
  }

  execution::Run input_failure{};
  const residency::ExecutionLease input_failure_lease =
      input_failure.begin(authority, one.plan);
  residency::ExecutionTicket failed_input{};
  if (!input_failure_lease ||
      !input_failure.issue(execution::Phase::Input, failed_input) ||
      !input_failure.terminal(failed_input,
                              residency::ExecutionTerminal::Failure,
                              Status::fail(Reason::TransferInvalid))) {
    return 35;
  }
  const residency::ExecutionClose input_failure_close =
      input_failure.close(520u);
  if (!input_failure_close || input_failure_close.success ||
      !input_failure_close.quarantined ||
      input_failure_close.failure_count != 1u ||
      input_failure_close.failures[0].phases != 1u ||
      input_failure_close.progress.native_dispatches != 0u ||
      !authority.probe(warm_keys, warm_host, 0u, 3u) ||
      !authority.probe(warm_keys, warm_device, 6u, 3u) ||
      warm_host != std::array<std::uint8_t, 3u>{0u, 0u, 0u} ||
      warm_device != std::array<std::uint8_t, 3u>{0u, 0u, 0u}) {
    return 36;
  }

  execution::Run stale_native{};
  const residency::ExecutionLease stale_lease =
      stale_native.begin(authority, one.plan);
  residency::ExecutionTicket stale_input{};
  execution::NativeEvidence stale_native_evidence = native;
  stale_native_evidence.token = stale_lease.token;
  stale_native_evidence.generation = stale_lease.generation + 1u;
  if (!stale_lease ||
      !stale_native.issue(execution::Phase::Input, stale_input) ||
      !stale_native.terminal(stale_input, residency::ExecutionTerminal::Success,
                             Status::success()) ||
      stale_native.native(stale_native_evidence)) {
    return 37;
  }
  stale_native_evidence.generation = stale_lease.generation;
  stale_native_evidence.native_dispatches = 0u;
  stale_native_evidence.native_completions = 0u;
  stale_native_evidence.native_inflight_peak = 0u;
  if (stale_native.native(stale_native_evidence)) {
    return 38;
  }
  stale_native_evidence.native_dispatches = 1u;
  stale_native_evidence.native_completions = 1u;
  stale_native_evidence.native_inflight_peak = 1u;
  residency::ExecutionTicket stale_output{};
  if (!stale_native.native(stale_native_evidence) ||
      !stale_native.issue(execution::Phase::Output, stale_output) ||
      !stale_native.terminal(stale_output,
                             residency::ExecutionTerminal::Success,
                             Status::success()) ||
      !stale_native.close(521u).success) {
    return 39;
  }

  execution::Run native_failure{};
  const residency::ExecutionLease native_failure_lease =
      native_failure.begin(authority, one.plan);
  residency::ExecutionTicket native_failure_input{};
  execution::NativeEvidence known_failure = native;
  known_failure.status = Status::fail(Reason::DeviceLost);
  known_failure.token = native_failure_lease.token;
  known_failure.generation = native_failure_lease.generation;
  known_failure.failures[0] = execution::FailureEvidence{
      .epoch = 0u, .phases = 2u, .may_write = 2u, .terminal = 2u};
  known_failure.failure_count = 1u;
  if (!native_failure_lease ||
      !native_failure.issue(execution::Phase::Input, native_failure_input) ||
      !native_failure.terminal(native_failure_input,
                               residency::ExecutionTerminal::Success,
                               Status::success()) ||
      !native_failure.native(known_failure)) {
    return 40;
  }
  const residency::ExecutionClose native_failure_close =
      native_failure.close(522u);
  if (!native_failure_close || native_failure_close.success ||
      !native_failure_close.quarantined ||
      native_failure_close.failure_count != 1u ||
      native_failure_close.failures[0].phases != 2u ||
      native_failure_close.progress.output_services != 0u) {
    return 41;
  }

  execution::Run native_unknown{};
  const residency::ExecutionLease unknown_lease =
      native_unknown.begin(authority, one.plan);
  residency::ExecutionTicket unknown_input{};
  execution::NativeEvidence unknown_native = known_failure;
  unknown_native.terminal = execution::TerminalKind::UnknownMayWrite;
  unknown_native.token = unknown_lease.token;
  unknown_native.generation = unknown_lease.generation;
  unknown_native.native_completions = 0u;
  unknown_native.failures[0].terminal = 0u;
  if (!unknown_lease ||
      !native_unknown.issue(execution::Phase::Input, unknown_input) ||
      !native_unknown.terminal(unknown_input,
                               residency::ExecutionTerminal::Success,
                               Status::success()) ||
      !native_unknown.native(unknown_native)) {
    return 42;
  }
  const residency::ExecutionClose unknown_native_close =
      native_unknown.close(523u);
  if (!unknown_native_close || unknown_native_close.success ||
      !unknown_native_close.quarantined ||
      unknown_native_close.failure_count != 1u ||
      unknown_native_close.failures[0].phases != 2u ||
      unknown_native_close.progress.native_completions != 0u ||
      !authority.probe(warm_keys, warm_host, 0u, 3u) ||
      !authority.probe(warm_keys, warm_device, 6u, 3u) ||
      warm_host != std::array<std::uint8_t, 3u>{0u, 0u, 0u} ||
      warm_device != std::array<std::uint8_t, 3u>{0u, 0u, 0u}) {
    return 43;
  }

  // A coordinator contradiction after all three physical owners terminated
  // cannot use exact failure evidence. Emergency abandon must conservatively
  // invalidate every Q=1 row and release the journal for a new admission.
  execution::Run abandoned{};
  const residency::ExecutionLease abandoned_lease =
      abandoned.begin(authority, one.plan);
  residency::ExecutionTicket abandoned_input{};
  residency::ExecutionTicket abandoned_output{};
  execution::NativeEvidence abandoned_native = native;
  abandoned_native.token = abandoned_lease.token;
  abandoned_native.generation = abandoned_lease.generation;
  if (!abandoned_lease ||
      !abandoned.issue(execution::Phase::Input, abandoned_input) ||
      !abandoned.terminal(abandoned_input,
                          residency::ExecutionTerminal::Success,
                          Status::success()) ||
      !abandoned.native(abandoned_native) ||
      !abandoned.issue(execution::Phase::Output, abandoned_output) ||
      !abandoned.terminal(abandoned_output,
                          residency::ExecutionTerminal::Success,
                          Status::success()) ||
      !abandoned.abandon() || abandoned.abandon() ||
      abandoned.close(524u).failure != residency::AuthorityFailure::Invalid ||
      !authority.probe(warm_keys, warm_host, 0u, 3u) ||
      !authority.probe(warm_keys, warm_device, 6u, 3u) ||
      warm_host != std::array<std::uint8_t, 3u>{0u, 0u, 0u} ||
      warm_device != std::array<std::uint8_t, 3u>{0u, 0u, 0u}) {
    return 44;
  }
  execution::Run abandoned_retry{};
  if (!abandoned_retry.begin(authority, one.plan) ||
      !abandoned_retry.abandon()) {
    return 44;
  }

  // Q=1 native owners address exact locals. A cached page at another local
  // must reject the compound admission before submit, and failed admission
  // must leave the prior cache authority untouched.
  residency::Authority permuted{};
  const auto owners = FrameOwners();
  for (std::size_t index = 0u; index < owners.size(); ++index) {
    std::uint32_t first_frame = 0u;
    if (!permuted.register_frames(owners[index].first, owners[index].second, 3u,
                                  first_frame) ||
        first_frame != index * 3u) {
      return 45;
    }
  }
  const residency::CacheUse misplaced_use{
      .key = warm_keys[0u],
      .access = residency::Access::Read,
      .next_use = 1u,
  };
  const residency::AuthorityResult misplaced = permuted.begin(
      std::span<const residency::CacheUse>{&misplaced_use, 1u},
      residency::FrameTier::Host, residency::FrameRole::Input, 1u, 1u);
  execution::Run rejected_run{};
  std::array<std::uint8_t, 1u> misplaced_resident{};
  const std::array<residency::CacheKey, 1u> misplaced_key{warm_keys[0u]};
  if (!misplaced || !permuted.activate(misplaced.lease.token) ||
      !permuted.complete(misplaced.lease.token, true) ||
      rejected_run.begin(permuted, one.plan).failure !=
          residency::AuthorityFailure::Capacity ||
      !permuted.probe(misplaced_key, misplaced_resident, 1u, 1u) ||
      misplaced_resident[0u] != 1u) {
    return 46;
  }

  const rund::compute::detail::DeviceOps &ops =
      rund::compute::detail::AccelDeviceOps();
  rund::compute::detail::DeviceState cpu{};
  execution::Owner unsupported{
      .native = std::make_shared<std::uint8_t>(1u),
      .host_bytes = 1u,
      .device_bytes = 2u,
      .staging_bytes = 3u,
  };
  std::uint64_t retained = 0u;
  if (cpu.ops != nullptr ||
      ops.residency.prepare_residency_selection == nullptr ||
      ops.residency.prepare_residency_execution == nullptr ||
      !unsupported.retained_bytes(retained) || retained != 6u ||
      ops.residency
              .prepare_residency_execution(
                  cpu,
                  std::span<const rund::node::accel::detail::
                                PreparedKernelPipeline *const>{},
                  unsupported)
              .reason() != Reason::BackendUnsupported ||
      unsupported || unsupported.host_bytes != 0u ||
      unsupported.device_bytes != 0u || unsupported.staging_bytes != 0u) {
    return 47;
  }
  return 0;
}

} // namespace rund_node_test_pipeline_residency::execution_test

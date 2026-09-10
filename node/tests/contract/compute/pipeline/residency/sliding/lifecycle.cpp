#include "local.hpp"

#include <array>
#include <memory>
#include <thread>
#include <utility>

namespace rund_node_test_pipeline_residency {
namespace {
namespace execution = sliding_detail::execution;
namespace residency = sliding_detail::residency;
using rund::compute::Reason;
using rund::compute::Status;
using rund::compute::detail::Type;
using sliding_detail::admit;
using sliding_detail::admit_bound;
using sliding_detail::direct_plan;
using sliding_detail::drain;
using sliding_detail::drain_bound;
using sliding_detail::EpochWork;
using sliding_detail::fetch;
using sliding_detail::fetch_bound;
using sliding_detail::region;
using sliding_detail::register_direct;
} // namespace

int CheckSlidingLifecycle() {

  if (execution::Sliding::create(
          execution::SlidingInvocation::direct(direct_plan(3u, 1u, 1u)), 100u,
          6u, 2u, 1u)) {
    return 1;
  }
  const std::shared_ptr<const execution::Plan> direct = direct_plan(5u, 1u);
  execution::Sliding sliding = execution::Sliding::create(
      execution::SlidingInvocation::direct(direct), 101u, 7u, 4u, 4u);
  if (!sliding) {
    return 1;
  }
  std::array<EpochWork, 5u> epochs{};
  execution::SlidingTicket stale{};
  for (std::size_t epoch = 0u; epoch < 3u; ++epoch) {
    if (!fetch(sliding, epoch, epochs[epoch], epoch == 0u ? &stale : nullptr) ||
        !admit(sliding, epochs[epoch]) ||
        !drain(sliding, epochs[epoch], true)) {
      return 2;
    }
  }
  // The input cell has been recycled with a new turn; a delayed old callback
  // cannot mutate the current slot.
  if (sliding.fetch_terminal(stale, Status::success(),
                             execution::TerminalKind::Known, true, 16u)) {
    return 3;
  }
  if (!fetch(sliding, 3u, epochs[3u]) || !admit(sliding, epochs[3u]) ||
      !fetch(sliding, 4u, epochs[4u]) || !admit(sliding, epochs[4u])) {
    return 4;
  }
  // e3 is deliberately not drained. e4 depends on the released bank0 e2,
  // not on a rigid whole-W4 Final, and is therefore admitted safely.
  execution::SlidingEvidence live{};
  if (!sliding.snapshot(live) || live.admitted != 5u ||
      live.terminal_frontier != 3u) {
    return 5;
  }
  if (!sliding.native_terminal(epochs[4u].native,
                               Status::fail(Reason::BackendFailed),
                               execution::TerminalKind::Known, false) ||
      !drain(sliding, epochs[3u], true)) {
    return 6;
  }
  execution::SlidingEvidence evidence{};
  if (!sliding.close_model(evidence) || !evidence.has_failure ||
      evidence.first_failure != execution::SlidingCoordinate{.ordinal = 4u,
                                                             .batch = 4u,
                                                             .stage = 0u} ||
      evidence.first_unsent != 5u || evidence.terminal_frontier != 5u ||
      evidence.quarantined || !sliding.quiescent()) {
    return 7;
  }

  // O=1 is per physical bank. An unpersisted e0 occupies bank0 but does not
  // block independent bank1 e1 Drain. The next bank0 epoch e2 is rejected
  // before native mutation until e0's exact Persist frees that local slot.
  execution::Sliding backpressure = execution::Sliding::create(
      execution::SlidingInvocation::direct(direct_plan(3u, 1u)), 102u, 8u, 2u,
      1u);
  EpochWork first{};
  EpochWork second{};
  EpochWork third{};
  execution::SlidingTicket first_output{};
  execution::SlidingTicket second_output{};
  if (!backpressure || !fetch(backpressure, 0u, first) ||
      !admit(backpressure, first) ||
      !drain(backpressure, first, false, &first_output) ||
      !fetch(backpressure, 1u, second) || !admit(backpressure, second) ||
      !drain(backpressure, second, false, &second_output) ||
      !fetch(backpressure, 2u, third) || admit(backpressure, third)) {
    return 8;
  }
  execution::SlidingTicket first_write{};
  execution::SlidingTicket second_write{};
  if (!backpressure.issue_persist(first_output, first_write) ||
      !backpressure.persist_terminal(first_write, Status::success(),
                                     execution::TerminalKind::Known, true,
                                     first_write.expected_bytes) ||
      !admit(backpressure, third) ||
      !backpressure.issue_persist(second_output, second_write) ||
      !backpressure.persist_terminal(second_write, Status::success(),
                                     execution::TerminalKind::Known, true,
                                     second_write.expected_bytes) ||
      !drain(backpressure, third, true) ||
      !backpressure.close_model(evidence)) {
    return 9;
  }
  execution::SlidingEvidence pressure{};
  if (!backpressure.snapshot(pressure) || pressure.admitted != 3u ||
      pressure.terminal_frontier != 3u || !backpressure.quiescent()) {
    return 10;
  }

  // Two concurrent backing lanes enter the model through one mutex-protected
  // terminal surface. Both exact tickets must become ready with no lost event.
  execution::Sliding concurrent = execution::Sliding::create(
      execution::SlidingInvocation::direct(direct_plan(2u, 2u)), 103u, 9u, 2u,
      2u);
  EpochWork pair{};
  if (!concurrent || !concurrent.project(0u, pair.uses, pair.projection) ||
      pair.projection.fetch_count != 2u) {
    return 11;
  }
  execution::SlidingTicket fetch0{};
  execution::SlidingTicket fetch1{};
  if (!concurrent.issue_fetch(pair.projection, pair.uses, 0u, fetch0) ||
      !concurrent.issue_fetch(pair.projection, pair.uses, 1u, fetch1)) {
    return 12;
  }
  bool terminal0 = false;
  bool terminal1 = false;
  std::thread lane0{[&] {
    terminal0 = concurrent.fetch_terminal(
        fetch0, Status::success(), execution::TerminalKind::Known, true, 16u);
  }};
  std::thread lane1{[&] {
    terminal1 = concurrent.fetch_terminal(
        fetch1, Status::success(), execution::TerminalKind::Known, true, 16u);
  }};
  lane0.join();
  lane1.join();
  if (!terminal0 || !terminal1 || !admit(concurrent, pair) ||
      !drain(concurrent, pair, true) || !concurrent.close_model(evidence)) {
    return 13;
  }

  // Logical Q does not change controller storage. One reusable work record
  // drives 100,000 coordinates through the same H=O=1 and W=4 owner.
  constexpr std::uint64_t Scale = 100'000u;
  execution::Sliding scale = execution::Sliding::create(
      execution::SlidingInvocation::direct(direct_plan(Scale, 1u)), 104u, 13u,
      1u, 1u);
  EpochWork scaled{};
  if (!scale) {
    return 13;
  }
  for (std::uint64_t epoch = 0u; epoch < Scale; ++epoch) {
    scaled = {};
    if (!fetch(scale, epoch, scaled) || !admit(scale, scaled) ||
        !drain(scale, scaled, true)) {
      return 14;
    }
  }
  if (!scale.close_model(evidence) || evidence.planned != Scale ||
      evidence.admitted != Scale || evidence.terminal_frontier != Scale ||
      evidence.fetch_calls != Scale || evidence.persist_calls != Scale) {
    return 15;
  }
  return 0;
}
} // namespace rund_node_test_pipeline_residency

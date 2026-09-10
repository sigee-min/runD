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

int CheckSlidingGraphProjection() {
  execution::SlidingEvidence evidence{};
  // Graph projection retains its ResidencyPlan owner and counts every
  // external output PageUse before native admission. Two writes cannot enter
  // a one-slot output ring and fail later after Device mutation.
  auto graph_fixture = sliding_detail::graph_fixture();
  if (graph_fixture.owner == nullptr) {
    return 16;
  }
  std::shared_ptr<const residency::ResidencyPlan> graph_owner =
      std::move(graph_fixture.owner);
  const std::array<std::uint64_t, 3u> graph_bytes = graph_fixture.bytes;
  execution::SlidingInvocation graph_invocation =
      execution::SlidingInvocation::graph(graph_owner, 1u, graph_bytes);
  graph_owner.reset();
  execution::Sliding graph_tight =
      execution::Sliding::create(graph_invocation, 105u, 11u, 1u, 1u);
  EpochWork graph_epoch{};
  if (graph_tight) {
    return 17;
  }
  execution::Sliding graph =
      execution::Sliding::create(graph_invocation, 106u, 12u, 1u, 2u);
  graph_epoch = {};
  if (!graph || !fetch(graph, 0u, graph_epoch) || !admit(graph, graph_epoch) ||
      !drain(graph, graph_epoch, true) || !graph.close_model(evidence) ||
      evidence.admitted != 1u || evidence.persist_calls != 2u) {
    return 18;
  }
  return 0;
}

int CheckSlidingGraphReplay() {
  execution::SlidingEvidence evidence{};
  // Direct H=1 is per physical bank. Bank1 may forecast its bank frontier e1
  // while current e0 owns bank0, but e3 cannot consume the slot reserved for
  // e1. Likewise, bank0 e2 cannot recycle e0 before its native terminal.
  execution::Sliding held = execution::Sliding::create(
      execution::SlidingInvocation::direct(direct_plan(4u, 1u)), 108u, 14u, 1u,
      1u);
  EpochWork held0{};
  EpochWork held1{};
  EpochWork held2{};
  EpochWork held3{};
  execution::SlidingTicket held_fetch0{};
  execution::SlidingTicket held_fetch1{};
  execution::SlidingTicket held_fetch2{};
  execution::SlidingTicket held_fetch3{};
  execution::SlidingTicket promote0{};
  if (!held || !held.project(0u, held0.uses, held0.projection) ||
      !held.issue_fetch(held0.projection, held0.uses, 0u, held_fetch0) ||
      !held.fetch_terminal(held_fetch0, Status::success(),
                           execution::TerminalKind::Known, true,
                           held_fetch0.expected_bytes) ||
      !held.project(3u, held3.uses, held3.projection) ||
      held.issue_fetch(held3.projection, held3.uses, 0u, held_fetch3) ||
      !held.project(1u, held1.uses, held1.projection) ||
      !held.issue_fetch(held1.projection, held1.uses, 0u, held_fetch1) ||
      !held.fetch_terminal(held_fetch1, Status::success(),
                           execution::TerminalKind::Known, true,
                           held_fetch1.expected_bytes) ||
      !held.project(2u, held2.uses, held2.projection) ||
      held.issue_fetch(held2.projection, held2.uses, 0u, held_fetch2) ||
      !held.issue_promote(held0.projection, held0.uses, promote0) ||
      !held.promote_terminal(promote0, Status::success(),
                             execution::TerminalKind::Known, true,
                             promote0.expected_bytes) ||
      !held.admit(held0.projection, held0.uses, held0.native) ||
      !drain(held, held0, true) ||
      !held.issue_fetch(held2.projection, held2.uses, 0u, held_fetch2) ||
      !held.fetch_terminal(held_fetch2, Status::success(),
                           execution::TerminalKind::Known, true,
                           held_fetch2.expected_bytes) ||
      !admit(held, held1) || !drain(held, held1, true) ||
      !held.issue_fetch(held3.projection, held3.uses, 0u, held_fetch3) ||
      !held.fetch_terminal(held_fetch3, Status::success(),
                           execution::TerminalKind::Known, true,
                           held_fetch3.expected_bytes) ||
      !admit(held, held2) || !drain(held, held2, true) || !admit(held, held3) ||
      !drain(held, held3, true) || !held.close_model(evidence)) {
    return 20;
  }

  // Tickets bind an internally minted owner nonce in addition to the same
  // Plan/token/generation. Isomorphic concurrent owners cannot replay them.
  const auto replay_plan = direct_plan(1u, 1u);
  execution::Sliding replay_a = execution::Sliding::create(
      execution::SlidingInvocation::direct(replay_plan), 109u, 15u, 1u, 1u);
  execution::Sliding replay_b = execution::Sliding::create(
      execution::SlidingInvocation::direct(replay_plan), 109u, 15u, 1u, 1u);
  EpochWork replay_work_a{};
  EpochWork replay_work_b{};
  execution::SlidingTicket replay_ticket_a{};
  execution::SlidingTicket replay_ticket_b{};
  if (!replay_a || !replay_b ||
      !replay_a.project(0u, replay_work_a.uses, replay_work_a.projection) ||
      !replay_b.project(0u, replay_work_b.uses, replay_work_b.projection) ||
      !replay_a.issue_fetch(replay_work_a.projection, replay_work_a.uses, 0u,
                            replay_ticket_a) ||
      !replay_b.issue_fetch(replay_work_b.projection, replay_work_b.uses, 0u,
                            replay_ticket_b) ||
      replay_a.fetch_terminal(replay_ticket_b, Status::success(),
                              execution::TerminalKind::Known, true,
                              replay_ticket_b.expected_bytes) ||
      replay_b.fetch_terminal(replay_ticket_a, Status::success(),
                              execution::TerminalKind::Known, true,
                              replay_ticket_a.expected_bytes) ||
      !replay_a.fetch_terminal(replay_ticket_a, Status::success(),
                               execution::TerminalKind::Known, true,
                               replay_ticket_a.expected_bytes) ||
      !replay_b.fetch_terminal(replay_ticket_b, Status::success(),
                               execution::TerminalKind::Known, true,
                               replay_ticket_b.expected_bytes) ||
      !admit(replay_a, replay_work_a) ||
      !drain(replay_a, replay_work_a, true) ||
      !admit(replay_b, replay_work_b) ||
      !drain(replay_b, replay_work_b, true) ||
      !replay_a.close_model(evidence) || !replay_b.close_model(evidence)) {
    return 21;
  }
  return 0;
}

int CheckSlidingGraphAdmission() {
  execution::SlidingEvidence evidence{};
  // Graph admission consumes a sealed topology predecessor. Stage 1 cannot
  // enter merely because its Host/output cells are available while stage 0
  // still owns the producer terminal.
  const residency::TiledGraphPlanInput wave_input{
      .page_count = 1u,
      .requested_frames = 1u,
      .max_frames = 1u,
      .prefetch_distance = 1u,
      .resources = {{.resource = 1u,
                     .type = Type::U64,
                     .page_bytes = 64u,
                     .logical_bytes = 64u,
                     .kind = residency::GraphResourceKind::ExternalInput,
                     .persistence = residency::ResourcePersistence::Backing},
                    {.resource = 2u,
                     .type = Type::U64,
                     .page_bytes = 64u,
                     .logical_bytes = 64u,
                     .kind = residency::GraphResourceKind::Internal,
                     .persistence = residency::ResourcePersistence::Transient},
                    {.resource = 3u,
                     .type = Type::U64,
                     .page_bytes = 64u,
                     .logical_bytes = 64u,
                     .kind = residency::GraphResourceKind::ExternalOutput,
                     .persistence = residency::ResourcePersistence::Transient}},
      .stages =
          {{.node = 1u,
            .ports = {{.resource = 1u, .access = residency::Access::Read},
                      {.resource = 2u, .access = residency::Access::Write}}},
           {.node = 2u,
            .ports = {{.resource = 2u, .access = residency::Access::Read},
                      {.resource = 3u, .access = residency::Access::Write}}}},
  };
  residency::PlanResult wave_plan = residency::PlanResidency(wave_input);
  if (!wave_plan) {
    return 26;
  }
  const std::shared_ptr<const residency::ResidencyPlan> wave_owner =
      std::make_shared<residency::ResidencyPlan>(std::move(wave_plan.plan));
  const std::array<std::uint64_t, 3u> wave_bytes{64u, 64u, 64u};
  execution::Sliding wave = execution::Sliding::create(
      execution::SlidingInvocation::graph(wave_owner, 1u, wave_bytes), 114u,
      20u, 1u, 1u);
  EpochWork stage0{};
  EpochWork stage1{};
  execution::SlidingTicket blocked_promote{};
  if (!wave || !fetch(wave, 0u, stage0) || !admit(wave, stage0) ||
      !wave.project(1u, stage1.uses, stage1.projection) ||
      wave.issue_promote(stage1.projection, stage1.uses, blocked_promote) ||
      !drain(wave, stage0, true) || !admit(wave, stage1) ||
      !drain(wave, stage1, true) || !wave.close_model(evidence)) {
    return 27;
  }
  return 0;
}

int CheckSlidingGraphTopology() {
  execution::SlidingEvidence evidence{};
  // Sealed Graph resource edges replace ordinal adjacency. Independent stage
  // 1 can be promoted while stage 0 still executes; join stage 2 waits for
  // both exact transient producers. Transients create no Host I/O.
  const residency::TiledGraphPlanInput branch_input{
      .page_count = 1u,
      .requested_frames = 1u,
      .max_frames = 1u,
      .prefetch_distance = 1u,
      .resources = {{.resource = 1u,
                     .type = Type::U64,
                     .page_bytes = 64u,
                     .logical_bytes = 64u,
                     .kind = residency::GraphResourceKind::ExternalInput,
                     .persistence = residency::ResourcePersistence::Backing},
                    {.resource = 2u,
                     .type = Type::U64,
                     .page_bytes = 128u,
                     .logical_bytes = 128u,
                     .kind = residency::GraphResourceKind::ExternalInput,
                     .persistence = residency::ResourcePersistence::Backing},
                    {.resource = 3u,
                     .type = Type::U64,
                     .page_bytes = 64u,
                     .logical_bytes = 64u,
                     .kind = residency::GraphResourceKind::Internal,
                     .persistence = residency::ResourcePersistence::Transient},
                    {.resource = 4u,
                     .type = Type::U64,
                     .page_bytes = 128u,
                     .logical_bytes = 128u,
                     .kind = residency::GraphResourceKind::Internal,
                     .persistence = residency::ResourcePersistence::Transient},
                    {.resource = 5u,
                     .type = Type::U64,
                     .page_bytes = 64u,
                     .logical_bytes = 64u,
                     .kind = residency::GraphResourceKind::ExternalOutput,
                     .persistence = residency::ResourcePersistence::Transient}},
      .stages =
          {{.node = 1u,
            .ports = {{.resource = 1u, .access = residency::Access::Read},
                      {.resource = 3u, .access = residency::Access::Write}}},
           {.node = 2u,
            .ports = {{.resource = 2u, .access = residency::Access::Read},
                      {.resource = 4u, .access = residency::Access::Write}}},
           {.node = 3u,
            .ports = {{.resource = 3u, .access = residency::Access::Read},
                      {.resource = 4u, .access = residency::Access::Read},
                      {.resource = 5u, .access = residency::Access::Write}}}},
  };
  residency::PlanResult branch_plan = residency::PlanResidency(branch_input);
  const std::array<std::uint64_t, 5u> branch_bytes{64u, 128u, 64u, 128u, 64u};
  if (!branch_plan) {
    return 39;
  }
  const std::shared_ptr<const residency::ResidencyPlan> branch_owner =
      std::make_shared<residency::ResidencyPlan>(std::move(branch_plan.plan));
  execution::Sliding branch = execution::Sliding::create(
      execution::SlidingInvocation::graph(branch_owner, 1u, branch_bytes), 127u,
      33u, 2u, 1u);
  EpochWork branch0{};
  EpochWork branch1{};
  EpochWork branch2{};
  execution::SlidingTicket blocked{};
  if (!branch || !fetch(branch, 0u, branch0) || !admit(branch, branch0) ||
      !fetch(branch, 1u, branch1) || !admit(branch, branch1) ||
      !branch.project(2u, branch2.uses, branch2.projection) ||
      branch.issue_promote(branch2.projection, branch2.uses, blocked) ||
      !drain(branch, branch1, true) ||
      branch.issue_promote(branch2.projection, branch2.uses, blocked) ||
      !drain(branch, branch0, true) || !admit(branch, branch2) ||
      !drain(branch, branch2, true) || !branch.close_model(evidence) ||
      evidence.fetch_calls != 2u || evidence.persist_calls != 1u) {
    return 40;
  }

  // If the later independent stage completes before the earlier stage fails,
  // its bounded terminal receipt remains Invalidating. The suffix Device
  // mutation cannot disappear when the contiguous frontier catches up.
  execution::Sliding branch_failure = execution::Sliding::create(
      execution::SlidingInvocation::graph(branch_owner, 1u, branch_bytes), 128u,
      34u, 2u, 1u);
  EpochWork failed_branch0{};
  EpochWork completed_branch1{};
  if (!branch_failure || !fetch(branch_failure, 0u, failed_branch0) ||
      !admit(branch_failure, failed_branch0) ||
      !fetch(branch_failure, 1u, completed_branch1) ||
      !admit(branch_failure, completed_branch1) ||
      !drain(branch_failure, completed_branch1, true) ||
      !branch_failure.native_terminal(failed_branch0.native,
                                      Status::fail(Reason::BackendFailed),
                                      execution::TerminalKind::Known, false) ||
      branch_failure.close_model(evidence) ||
      !branch_failure.invalidate(completed_branch1.native) ||
      !branch_failure.close_model(evidence) ||
      evidence.terminal_frontier != 2u) {
    return 41;
  }
  return 0;
}
} // namespace rund_node_test_pipeline_residency

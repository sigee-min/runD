#include "local.hpp"

#include <thread>

namespace rund_node_test_pipeline_residency {
namespace {

namespace execution = sliding_detail::execution;
namespace residency = sliding_detail::residency;
using sliding_detail::EpochWork;
using sliding_detail::admit;
using sliding_detail::admit_bound;
using sliding_detail::direct_plan;
using sliding_detail::drain;
using sliding_detail::drain_bound;
using sliding_detail::fetch;
using sliding_detail::fetch_bound;
using sliding_detail::register_direct;
using sliding_detail::region;
using rund::compute::Reason;
using rund::compute::Status;
using rund::compute::detail::Type;

} // namespace

int CheckSlidingGeneration() {
  execution::SlidingEvidence evidence{};

  // The product final is a true two-party freeze. Authority mints the unique
  // run owner, Sliding freezes every terminal ingress, Authority freezes the
  // physical generation without changing frames, and only the subsequent
  // accept retires that generation and yields a one-use controller receipt.
  constexpr std::uint32_t FinalK = 1u;
  constexpr std::uint32_t FinalH = 4u;
  const std::shared_ptr<const execution::Plan> final_plan =
      direct_plan(5u, FinalK, FinalH);
  residency::Authority final_authority{};
  residency::Authority other_authority{};
  auto final_sliding = final_authority.sliding();
  auto other_sliding = other_authority.sliding();
  if (final_plan == nullptr ||
      !register_direct(final_authority, FinalK, FinalH) ||
      !register_direct(other_authority, FinalK, FinalH)) {
    return 42;
  }
  const residency::CacheUse prior{
      .key = {.backing = 77u, .version = 1u, .page = 0u},
      .access = residency::Access::Read,
      .next_use = residency::NeverUse,
  };
  const residency::AuthorityResult seeded = final_authority.begin(
      std::span<const residency::CacheUse>{&prior, 1u},
      residency::FrameTier::Host, residency::FrameRole::Input, 0u, FinalH);
  if (!seeded || !final_authority.complete(seeded.lease.token, true)) {
    return 43;
  }
  const residency::ExecutionLease final_lease =
      final_sliding.begin_execution_sliding(*final_plan);
  const residency::ExecutionLease other_lease =
      other_sliding.begin_execution_sliding(*final_plan);
  execution::Sliding final_run = execution::Sliding::create_bound(
      execution::SlidingInvocation::direct(final_plan), final_lease.token,
      final_lease.generation, FinalH, FinalH);
  execution::Sliding duplicate_run = execution::Sliding::create_bound(
      execution::SlidingInvocation::direct(final_plan), final_lease.token,
      final_lease.generation, FinalH, FinalH);
  execution::Sliding late_bind_run = execution::Sliding::create_bound(
      execution::SlidingInvocation::direct(final_plan), final_lease.token,
      final_lease.generation, FinalH, FinalH);
  EpochWork late_bind_work{};
  std::array<EpochWork, 5u> final_work{};
  if (!final_lease || !other_lease ||
      final_lease.generation == other_lease.generation || !final_run ||
      !duplicate_run || !late_bind_run ||
      fetch(late_bind_run, 0u, late_bind_work) ||
      !final_sliding.bind_execution_sliding(final_run) ||
      final_sliding.bind_execution_sliding(duplicate_run) ||
      final_sliding.abandon_execution_sliding(*final_plan, final_lease)) {
    return 44;
  }
  for (std::uint64_t epoch = 0u; epoch < final_work.size(); ++epoch) {
    if (!fetch_bound(final_authority, *final_plan, final_run, epoch,
                     final_work[epoch]) ||
        !admit_bound(final_authority, *final_plan, final_run,
                     final_work[epoch]) ||
        !drain_bound(final_authority, *final_plan, final_run, final_work[epoch],
                     true)) {
      return 45;
    }
  }
  execution::SlidingFinal frozen{};
  residency::ExecutionSlidingFinal physical{};
  residency::ExecutionSlidingReceipt accepted{};
  if (!final_run.prepare_final(frozen) ||
      other_sliding.prepare_execution_sliding(*final_plan, frozen,
                                                physical) ||
      !final_sliding.prepare_execution_sliding(*final_plan, frozen,
                                                 physical) ||
      !physical || final_run.close_model(evidence)) {
    return 46;
  }
  const residency::ExecutionClose final_closed =
      final_sliding.accept_execution_sliding(std::move(physical), accepted);
  residency::ExecutionSlidingFinal stale_final{};
  const residency::ExecutionClose stale_closed =
      final_sliding.accept_execution_sliding(std::move(stale_final), accepted);
  execution::SlidingEvidence replay_evidence{};
  if (!final_closed || !final_closed.success ||
      stale_closed.failure != residency::AuthorityFailure::Invalid ||
      stale_closed.success || stale_closed.quarantined || !accepted ||
      !final_run.accept_final(frozen, std::move(accepted), evidence) ||
      final_run.accept_final(frozen, std::move(accepted), replay_evidence) ||
      evidence.admitted != 5u || evidence.has_failure) {
    return 47;
  }
  std::array<std::uint8_t, 1u> present{1u};
  const std::array<residency::CacheKey, 1u> prior_key{prior.key};
  const std::array<residency::CacheKey, 1u> final_page_key{
      residency::CacheKey{.backing = 11u, .version = 3u, .page = 4u}};
  std::array<std::uint8_t, 1u> final_page_present{0u};
  if (!final_authority.probe(prior_key, present, 0u, FinalH) ||
      present[0] != 0u ||
      !final_authority.probe(final_page_key, final_page_present, 0u, FinalH) ||
      final_page_present[0] != 1u ||
      !other_sliding.abandon_execution_sliding(*final_plan, other_lease)) {
    return 48;
  }

  // An already-returned suffix Native owns only generation-private Device
  // output.  When an earlier Native fails, whole-generation abort cancels the
  // callback-free Reserved outputs without issuing meaningless D2H/Persist.
  residency::Authority suffix_authority{};
  auto suffix_sliding = suffix_authority.sliding();
  if (!register_direct(suffix_authority, FinalK, FinalH)) {
    return 84;
  }
  const residency::ExecutionLease suffix_lease =
      suffix_sliding.begin_execution_sliding(*final_plan);
  execution::Sliding suffix_run = execution::Sliding::create_bound(
      execution::SlidingInvocation::direct(final_plan), suffix_lease.token,
      suffix_lease.generation, FinalH, FinalH);
  EpochWork suffix0{};
  EpochWork suffix1{};
  execution::SlidingFinal suffix_frozen{};
  residency::ExecutionSlidingFinal suffix_physical{};
  residency::ExecutionSlidingReceipt suffix_receipt{};
  if (!suffix_lease || !suffix_run ||
      !suffix_sliding.bind_execution_sliding(suffix_run) ||
      !fetch_bound(suffix_authority, *final_plan, suffix_run, 0u, suffix0) ||
      !admit_bound(suffix_authority, *final_plan, suffix_run, suffix0) ||
      !fetch_bound(suffix_authority, *final_plan, suffix_run, 1u, suffix1) ||
      !admit_bound(suffix_authority, *final_plan, suffix_run, suffix1) ||
      !suffix_sliding.terminal_execution_sliding_native(
          suffix_run, *suffix1.physical_native, Status::success(),
          execution::TerminalKind::Known, true) ||
      !suffix_sliding.release_execution_sliding_native(
          suffix_run, std::move(*suffix1.physical_native)) ||
      !suffix_sliding.terminal_execution_sliding_native(
          suffix_run, *suffix0.physical_native,
          Status::fail(Reason::BackendFailed), execution::TerminalKind::Known,
          true) ||
      !suffix_sliding.release_execution_sliding_native(
          suffix_run, std::move(*suffix0.physical_native)) ||
      !suffix_run.prepare_final(suffix_frozen) ||
      suffix_frozen.evidence().drain_calls != 0u ||
      !suffix_sliding.prepare_execution_sliding(*final_plan, suffix_frozen,
                                                  suffix_physical) ||
      !suffix_sliding.accept_execution_sliding(std::move(suffix_physical),
                                                 suffix_receipt) ||
      !suffix_run.accept_final(suffix_frozen, std::move(suffix_receipt),
                               evidence) ||
      !evidence.has_failure || evidence.admitted != 2u ||
      evidence.terminal_frontier != 1u) {
    return 85;
  }
  const residency::ExecutionLease suffix_retry =
      suffix_sliding.begin_execution_sliding(*final_plan);
  if (!suffix_retry ||
      !suffix_sliding.abandon_execution_sliding(*final_plan, suffix_retry)) {
    return 86;
  }

  // A Known may-write terminal does not require callback-order-dependent
  // per-cell invalidation. The frozen Authority generation abort covers the
  // whole fixed reservation and the same Pool is immediately reusable.
  residency::Authority failed_authority{};
  auto failed_sliding = failed_authority.sliding();
  if (!register_direct(failed_authority, FinalK, FinalH)) {
    return 49;
  }
  const residency::ExecutionLease failed_lease =
      failed_sliding.begin_execution_sliding(*final_plan);
  execution::Sliding failed_run = execution::Sliding::create_bound(
      execution::SlidingInvocation::direct(final_plan), failed_lease.token,
      failed_lease.generation, FinalH, FinalH);
  EpochWork failed_work{};
  execution::SlidingFinal failed_frozen{};
  residency::ExecutionSlidingFinal failed_physical{};
  residency::ExecutionSlidingReceipt failed_receipt{};
  if (!failed_lease || !failed_run ||
      !failed_sliding.bind_execution_sliding(failed_run) ||
      !fetch_bound(failed_authority, *final_plan, failed_run, 0u,
                   failed_work) ||
      !admit_bound(failed_authority, *final_plan, failed_run, failed_work) ||
      !failed_sliding.terminal_execution_sliding_native(
          failed_run, *failed_work.physical_native,
          Status::fail(Reason::BackendFailed), execution::TerminalKind::Known,
          true) ||
      !failed_sliding.release_execution_sliding_native(
          failed_run, std::move(*failed_work.physical_native)) ||
      failed_run.close_model(evidence) ||
      !failed_run.prepare_final(failed_frozen) ||
      !failed_sliding.prepare_execution_sliding(*final_plan, failed_frozen,
                                                  failed_physical)) {
    return 50;
  }
  const residency::ExecutionClose failed_closed =
      failed_sliding.accept_execution_sliding(std::move(failed_physical),
                                                failed_receipt);
  if (!failed_closed || failed_closed.success || !failed_receipt ||
      !failed_run.accept_final(failed_frozen, std::move(failed_receipt),
                               evidence) ||
      !evidence.has_failure || !failed_run.quiescent()) {
    return 51;
  }
  const residency::ExecutionLease retry_lease =
      failed_sliding.begin_execution_sliding(*final_plan);
  if (!retry_lease ||
      !failed_sliding.abandon_execution_sliding(*final_plan, retry_lease)) {
    return 52;
  }

  // Dirty sole-newer data is never admitted into a destructive Sliding
  // generation. Admission fails before any controller exists.
  residency::Authority dirty_authority{};
  auto dirty_sliding = dirty_authority.sliding();
  if (!register_direct(dirty_authority, FinalK, FinalH)) {
    return 53;
  }
  const residency::CacheUse dirty{
      .key = {.backing = 88u, .version = 1u, .page = 0u},
      .access = residency::Access::ReadWrite,
      .next_use = residency::NeverUse,
      .dirty = {.bytes = 16u},
  };
  const residency::AuthorityResult dirtied = dirty_authority.begin(
      std::span<const residency::CacheUse>{&dirty, 1u},
      residency::FrameTier::Host, residency::FrameRole::Input, 0u, FinalH);
  if (!dirtied || !dirty_authority.complete(dirtied.lease.token, true) ||
      dirty_sliding.begin_execution_sliding(*final_plan)) {
    return 54;
  }

  // A clean HostReady row survives Promote and is rebound by exact PageKey,
  // next-use, and pin facts. Two Graph consumers of the same external page
  // perform one backing Fetch; the second consumer is an authenticated hit.
  const residency::TiledGraphPlanInput reuse_input{
      .page_count = 1u,
      .requested_frames = 1u,
      .max_frames = 1u,
      .prefetch_distance = 2u,
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
            .ports = {{.resource = 1u, .access = residency::Access::Read},
                      {.resource = 2u, .access = residency::Access::Read},
                      {.resource = 3u, .access = residency::Access::Write}}}},
  };
  residency::PlanResult reuse_plan = residency::PlanResidency(reuse_input);
  if (!reuse_plan) {
    return 55;
  }
  const auto reuse_owner = std::make_shared<const residency::ResidencyPlan>(
      std::move(reuse_plan.plan));
  const std::array<std::uint64_t, 3u> reuse_bytes{64u, 64u, 64u};
  execution::Sliding reuse = execution::Sliding::create(
      execution::SlidingInvocation::graph(reuse_owner, 1u, reuse_bytes), 131u,
      37u, 1u, 1u);
  EpochWork reuse0{};
  EpochWork reuse1{};
  if (!reuse || !fetch(reuse, 0u, reuse0) ||
      !reuse.project(1u, reuse1.uses, reuse1.projection) ||
      reuse.reuse_fetch(reuse1.projection, reuse1.uses, 0u) ||
      !admit(reuse, reuse0) || !drain(reuse, reuse0, false) ||
      !reuse.reuse_fetch(reuse1.projection, reuse1.uses, 0u) ||
      !admit(reuse, reuse1) || !drain(reuse, reuse1, true) ||
      !reuse.close_model(evidence) || evidence.fetch_calls != 1u ||
      evidence.fetch_hits != 1u || evidence.promote_calls != 2u ||
      evidence.persist_calls != 1u) {
    return 56;
  }

  // With both clean rows populated, the bounded Belady scan for the current
  // coordinate replaces the unpinned resident whose sealed next-use is
  // farthest (slot 1 here), with physical slot index as the deterministic tie
  // break. The H=1 `held` case above separately proves that a same-bank future
  // miss may not replace the current unconsumed Ready row.
  execution::Sliding victim = execution::Sliding::create(
      execution::SlidingInvocation::direct(direct_plan(5u, 1u, 2u)), 132u, 38u,
      2u, 2u);
  std::array<EpochWork, 5u> victim_work{};
  execution::SlidingTicket victim_fetch0{};
  execution::SlidingTicket victim_fetch2{};
  for (std::size_t epoch = 0u; epoch < 4u; ++epoch) {
    execution::SlidingTicket *const captured = epoch == 0u   ? &victim_fetch0
                                               : epoch == 2u ? &victim_fetch2
                                                             : nullptr;
    if (!victim || !fetch(victim, epoch, victim_work[epoch], captured) ||
        !admit(victim, victim_work[epoch]) ||
        !drain(victim, victim_work[epoch], true)) {
      return 57;
    }
  }
  execution::SlidingTicket victim_fetch4{};
  if (!victim.project(4u, victim_work[4u].uses, victim_work[4u].projection) ||
      !victim.issue_fetch(victim_work[4u].projection, victim_work[4u].uses, 0u,
                          victim_fetch4) ||
      victim_fetch4.slot != victim_fetch2.slot ||
      victim_fetch4.slot == victim_fetch0.slot ||
      !victim.fetch_terminal(victim_fetch4, Status::success(),
                             execution::TerminalKind::Known, true,
                             victim_fetch4.expected_bytes) ||
      !admit(victim, victim_work[4u]) ||
      !drain(victim, victim_work[4u], true) || !victim.close_model(evidence) ||
      evidence.fetch_calls != 5u || evidence.fetch_hits != 0u) {
    return 57;
  }

  // A centered Window whose raw input is exactly four canonical P=12 pages
  // still has a clipped expanded frame tail: N=48, F=16, R=2 leaves the
  // canonical extent at zero while the expanded extent is N. Seal validates
  // those identities independently and keeps the full canonical last page.
  const auto exact_window = direct_plan(4u, 2u, 4u, residency::NeverUse, 192u,
                                        execution::FetchFill::RepeatBoundary,
                                        64u, 48u, 8u, 8u, 4u, true, 48u, 0u);
  const auto wrong_canonical_extent =
      direct_plan(4u, 2u, 4u, residency::NeverUse, 192u,
                  execution::FetchFill::RepeatBoundary, 64u, 48u, 8u, 8u, 4u,
                  true, 48u, 48u);
  const auto wrong_expanded_extent = direct_plan(
      4u, 2u, 4u, residency::NeverUse, 192u,
      execution::FetchFill::RepeatBoundary, 64u, 48u, 8u, 8u, 4u, true, 0u, 0u);
  execution::FetchSource expanded_tail{};
  execution::FetchSource canonical_tail{};
  execution::WindowFootprintProjection footprint{};
  if (!exact_window || wrong_canonical_extent || wrong_expanded_extent ||
      !exact_window->input_source(3u, expanded_tail) ||
      expanded_tail.bytes != 56u || expanded_tail.frame_bytes != 64u ||
      !exact_window->canonical_input_source(3u, canonical_tail) ||
      canonical_tail.bytes != 48u || canonical_tail.frame_bytes != 48u ||
      canonical_tail.fill != execution::FetchFill::None ||
      !exact_window->window_footprint(1u, footprint) ||
      footprint.target_count != 2u || footprint.source_count != 3u) {
    return 58;
  }
  return 0;
}

} // namespace rund_node_test_pipeline_residency

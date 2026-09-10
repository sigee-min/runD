#include "final.hpp"

#include "cache.hpp"
#include "evidence.hpp"

#include "../../device/residency/pool.hpp"
#include "../../pipeline/claim.hpp"

namespace rund::compute::detail {

namespace {

[[nodiscard]] Status isolate_stage(VirtualRunResources &resources,
                                   bool &poison_pipeline) noexcept {
  const Status isolated = quarantine_pool_stage(resources);
  poison_pipeline = true;
  return resources.view_commit == nullptr
             ? Status::fail(Reason::DeviceLost)
             : isolated;
}

} // namespace

Status finish_virtual_run(VirtualPipelineState &state, Stats stats,
                          VirtualRunTransaction &transaction, Status status,
                          const VirtualRunWriteCertainty certainty,
                          const std::uint64_t failed_page,
                          const std::uint64_t output_hash,
                          bool &poison_pipeline,
                          VirtualRunResources *const resources) noexcept {
  VirtualRunPublicationCursor cursor = transaction.cursor;
  const auto publication = [](const VirtualRunPublicationCursor &value,
                              const std::size_t bank) noexcept {
    return DeferredPipelinePublication{
        .base_generation = value.base_generation[bank],
        .base_payload_epoch = value.base_payload_epoch[bank],
        .control_generation = value.control_generation[bank],
        .terminal_count = value.terminal_count[bank],
        .base_parity = value.base_parity[bank]};
  };
  const auto restore = [&]() noexcept {
    if (!cursor.active) {
      return Status::success();
    }
    const Status restored =
        abort_virtual_run_publication_cursor(state, cursor);
    if (!restored) {
      poison_pipeline = true;
    }
    return restored;
  };

  Status committed = status;
  VirtualRunWriteCertainty effective_certainty = certainty;
  if (resources != nullptr && resources->view_commit != nullptr) {
    committed = isolate_stage(*resources, poison_pipeline);
    effective_certainty = VirtualRunWriteCertainty::UnknownMayWrite;
  }
  if (transaction.started) {
    const auto abort = [&](const VirtualRunWriteCertainty requested,
                           const Status failure) noexcept {
      Status outcome = failure;
      abort_virtual_run_transaction(state, transaction, requested,
                                    effective_certainty, outcome,
                                    poison_pipeline);
      committed = outcome;
    };
    if (effective_certainty == VirtualRunWriteCertainty::UnknownMayWrite) {
      abort(VirtualRunWriteCertainty::UnknownMayWrite,
            Status::fail(Reason::DeviceLost));
    } else if (!status) {
      abort(effective_certainty, status);
    } else {
      const Status prepared =
          prepare_virtual_run_transaction(state, transaction);
      if (!prepared) {
        abort(VirtualRunWriteCertainty::KnownNoWrite, prepared);
      } else if (cursor.active &&
                 (!state.pipeline || !state.alternate_pipeline)) {
        abort(VirtualRunWriteCertainty::KnownNoWrite,
              Status::fail(Reason::PipelineInvalid));
      } else {
        Status ready = Status::success();
        if (cursor.active) {
          ready = preflight_deferred_pipeline_generations(
              *state.pipeline, publication(cursor, 0u),
              *state.alternate_pipeline, publication(cursor, 1u));
        }
        if (!ready) {
          abort(VirtualRunWriteCertainty::KnownNoWrite, ready);
        } else if (transaction.scan) {
          const Status rows =
              prepare_virtual_run_transaction_rows(state, transaction, true);
          if (!rows) {
            abort(VirtualRunWriteCertainty::UnknownMayWrite, rows);
          } else {
            committed = commit_virtual_run_transaction(state, transaction,
                                                       poison_pipeline,
                                                       effective_certainty);
            if (committed && cursor.active) {
              apply_deferred_pipeline_generations(
                  *state.pipeline, publication(cursor, 0u),
                  *state.alternate_pipeline, publication(cursor, 1u));
            }
          }
        } else {
          committed = commit_virtual_run_transaction(state, transaction,
                                                     poison_pipeline,
                                                     effective_certainty);
          if (committed && cursor.active) {
            apply_deferred_pipeline_generations(
                *state.pipeline, publication(cursor, 0u),
                *state.alternate_pipeline, publication(cursor, 1u));
          }
        }
      }
    }
  } else if (committed && cursor.active) {
    if (!state.pipeline || !state.alternate_pipeline) {
      committed = Status::fail(Reason::PipelineInvalid);
      poison_pipeline = true;
    } else {
      const Status ready = preflight_deferred_pipeline_generations(
          *state.pipeline, publication(cursor, 0u), *state.alternate_pipeline,
          publication(cursor, 1u));
      if (!ready) {
        committed = ready;
      } else {
        apply_deferred_pipeline_generations(
            *state.pipeline, publication(cursor, 0u),
            *state.alternate_pipeline, publication(cursor, 1u));
      }
    }
  }
  if (cursor.active && effective_certainty ==
                          VirtualRunWriteCertainty::UnknownMayWrite) {
    const Status poisoned =
        poison_virtual_run_publication_cursor(state, cursor);
    poison_pipeline = true;
    if (committed && !poisoned) {
      committed = Status::fail(Reason::DeviceLost);
    }
  }
  if (cursor.active && !committed &&
      effective_certainty == VirtualRunWriteCertainty::KnownNoWrite) {
    const Status restored = restore();
    if (!restored) {
      committed = restored;
    }
  }
  return publish_virtual_run_evidence(state, std::move(stats), committed,
                                      failed_page, output_hash,
                                      poison_pipeline);
}

Status finish_virtual_empty(VirtualPipelineState &state, Stats stats,
                            VirtualRunTransaction &transaction,
                            const VirtualRunProjection &run,
                            VirtualBacking &output, VirtualReduction &reduction,
                            std::uint64_t &failed_page,
                            bool &poison_pipeline,
                            VirtualRunResources *const resources) noexcept {
  if (resources != nullptr && resources->view_commit != nullptr) {
    const Status isolated = isolate_stage(*resources, poison_pipeline);
    return finish_virtual_run(
        state, std::move(stats), transaction, isolated,
        VirtualRunWriteCertainty::UnknownMayWrite, failed_page, 0u,
        poison_pipeline, resources);
  }
  ::rund::node::hash_detail::Fnv output_hash{};
  if (run.reduction()) {
    const Status reduced = finish_virtual_reduction(
        output, run, reduction, stats.pipeline.residency, output_hash);
    if (!reduced) {
      return finish_virtual_run(state, std::move(stats), transaction, reduced,
                                VirtualRunWriteCertainty::KnownNoWrite,
                                failed_page, 0u, poison_pipeline, resources);
    }
    clear_virtual_recovery(output);
  }
  return finish_virtual_run(state, std::move(stats), transaction,
                            Status::success(),
                            VirtualRunWriteCertainty::KnownNoWrite, failed_page,
                            output_hash.Finish(), poison_pipeline, resources);
}

Status finish_virtual_dispatch(VirtualPipelineState &state, Stats stats,
                               VirtualRunTransaction &transaction,
                               const VirtualRunProjection &run,
                               VirtualBacking &output,
                               VirtualReduction &reduction,
                               const VirtualRunDispatchResult &dispatched,
                               std::uint64_t &failed_page,
                               bool &poison_pipeline,
                               VirtualRunResources *const resources) noexcept {
  if (resources != nullptr && resources->view_commit != nullptr) {
    const Status isolated = isolate_stage(*resources, poison_pipeline);
    return finish_virtual_run(
        state, std::move(stats), transaction, isolated,
        VirtualRunWriteCertainty::UnknownMayWrite, failed_page, 0u,
        poison_pipeline, resources);
  }
  if (dispatched.failed_page != ResidencyStats::no_failed_page) {
    failed_page = dispatched.failed_page;
  }
  poison_pipeline = dispatched.poison_pipeline;
  if (!dispatched.status) {
    return finish_virtual_run(state, std::move(stats), transaction,
                              dispatched.status, dispatched.certainty,
                              failed_page, dispatched.output_hash,
                              poison_pipeline, resources);
  }

  ::rund::node::hash_detail::Fnv output_hash{};
  if (dispatched.reduction_pending) {
    const Status reduced = finish_virtual_reduction(
        output, run, reduction, stats.pipeline.residency, output_hash);
    if (!reduced) {
      failed_page = run.graph_execution()
                        ? run.active.graph.page_count() - 1u
                        : (run.active.stream.page_count() == 0u
                               ? ResidencyStats::no_failed_page
                               : run.active.stream.page_count() - 1u);
      return finish_virtual_run(state, std::move(stats), transaction, reduced,
                                VirtualRunWriteCertainty::KnownNoWrite,
                                failed_page, 0u, poison_pipeline, resources);
    }
    clear_virtual_recovery(output);
  } else if (dispatched.selected || dispatched.direct_terminal ||
             dispatched.graph_execution) {
    return finish_virtual_run(state, std::move(stats), transaction,
                              Status::success(), dispatched.certainty,
                              failed_page, dispatched.output_hash,
                              poison_pipeline, resources);
  } else {
    if (state.pipeline->residency_pool == nullptr) {
      poison_pipeline = true;
      return finish_virtual_run(state, std::move(stats), transaction,
                                Status::fail(Reason::PipelineInvalid),
                                VirtualRunWriteCertainty::UnknownMayWrite,
                                failed_page, 0u, poison_pipeline, resources);
    }
    residency::Pool &pool = *state.pipeline->residency_pool;
    const residency::AuthorityResult drain = pool.authority().drain_dirty(
        pool.first_output_frame, pool.output_frame_count);
    if (!drain) {
      poison_pipeline = true;
      return finish_virtual_run(state, std::move(stats), transaction,
                                Status::fail(Reason::PipelineInvalid),
                                VirtualRunWriteCertainty::UnknownMayWrite,
                                failed_page, 0u, poison_pipeline, resources);
    }
    const Status written = writeback_residency_cache(
        output, run, drain.lease.transitions, stats.pipeline.residency,
        nullptr, {}, &transaction);
    const bool published =
        written && pool.authority().complete(drain.lease.token, true);
    bool completed =
        written ? published : pool.authority().discard(drain.lease.token);
    if (written && !published) {
      completed = pool.authority().discard(drain.lease.token);
    }
    if (!written || !published || !completed) {
      failed_page = drain.lease.transitions.empty()
                        ? ResidencyStats::no_failed_page
                        : drain.lease.transitions.front().key.page;
      poison_pipeline = (written && !published) || !completed;
      return finish_virtual_run(state, std::move(stats), transaction,
                                written ? Status::fail(Reason::PipelineInvalid)
                                        : written,
                                VirtualRunWriteCertainty::UnknownMayWrite,
                                failed_page, 0u, poison_pipeline, resources);
    }
    if (!transaction.started) {
      clear_virtual_recovery(output);
    }
  }

  return finish_virtual_run(
      state, std::move(stats), transaction, Status::success(),
      dispatched.certainty, failed_page,
      dispatched.reduction_pending ? output_hash.Finish()
                                   : dispatched.output_hash,
      poison_pipeline, resources);
}

} // namespace rund::compute::detail

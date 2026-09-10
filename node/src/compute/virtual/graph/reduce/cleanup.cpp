#include "cleanup.hpp"

#include "authority.hpp"
#include "failure.hpp"
#include "promote.hpp"

#include "../../../device/residency/registry/graph_drain_owner.hpp"
#include "../../../device/residency/registry/graph_persist_owner.hpp"

#include <array>
#include <span>
#include <utility>

namespace rund::compute::detail::graph_reduce {

bool cleanup_ticket(residency::Authority &authority, Ticket &ticket,
                    FailLog &failure) noexcept {
  const auto report =
      [&](const Status status, const Check check,
          const std::uint64_t epoch = Fail::NoEpoch,
          const std::uint64_t token = 0u, const std::uint64_t generation = 0u,
          const residency::CloseInfo *const info = nullptr) noexcept {
        if (failure.has()) {
          if (info != nullptr) {
            const Fail primary = failure.first();
            failure.attach(primary.stage, primary.batch, Phase::Recovery,
                           Check::Recover, primary.epoch, *info);
          }
          return;
        }
        failure.note_cred(Fail::NoStage, ticket.batch, Phase::Recovery, check,
                          status, epoch, token, generation, info);
      };
  bool clean = true;
  auto drains = authority.graph_drains();
  auto persists = authority.graph_persists();
  if (ticket.input_promote) {
    const bool terminal = cancel_input_promotion(authority, ticket);
    clean = terminal && clean;
    if (terminal) {
      ticket.prefix_token = 0u;
    } else {
      report(Status::fail(Reason::PipelineBusy), Check::Recover,
             ticket.prefix_epoch.ordinal, ticket.prefix_receipt.token(),
             ticket.prefix_receipt.generation());
    }
  }
  if (ticket.output_drain) {
    std::array<residency::execution::GraphDrainCompletion,
               residency::execution::GraphDrainCapacity>
        completions{};
    const auto pages = ticket.output_drain.pages();
    for (std::size_t index = 0u; index < pages.size(); ++index) {
      completions[index] = residency::execution::GraphDrainCompletion{
          .key = pages[index].key,
          .bytes = 0u,
          .source_frame = pages[index].source_frame,
          .target_frame = pages[index].target_frame,
      };
    }
    (void)drains.terminal_graph_drain(
        ticket.output_drain, Status::fail(Reason::CompletionInvalid),
        residency::execution::TerminalKind::UnknownMayWrite, true,
        std::span<const residency::execution::GraphDrainCompletion>{
            completions.data(), pages.size()});
    const bool terminal =
        drains.release_graph_drain(std::move(ticket.output_drain));
    clean = terminal && clean;
    if (terminal) {
      ticket.output_dirty = false;
    } else {
      report(Status::fail(Reason::PipelineBusy), Check::Recover, Fail::NoEpoch);
    }
  }
  if (ticket.collective_token != 0u) {
    const bool cpu = ticket.collective != nullptr &&
                     ticket.collective->device != nullptr &&
                     ticket.collective->device->backend == Backend::Cpu;
    residency::CloseInfo info{};
    const bool terminal =
        cpu ? close_cpu_epoch(authority, ticket.collective_receipt, false, true,
                              &info)
            : authority.complete(ticket.collective_token, false, true);
    clean = terminal && clean;
    if (terminal) {
      ticket.collective_token = 0u;
      ticket.output_dirty = false;
    } else {
      report(Status::fail(Reason::PipelineBusy), Check::Recover,
             ticket.collective_epoch.ordinal, ticket.collective_receipt.token(),
             ticket.collective_receipt.generation(), cpu ? &info : nullptr);
    }
  }
  if (ticket.prefix_token != 0u) {
    const bool cpu = ticket.prefix != nullptr &&
                     ticket.prefix->device != nullptr &&
                     ticket.prefix->device->backend == Backend::Cpu;
    residency::CloseInfo info{};
    const bool terminal =
        cpu ? close_cpu_epoch(authority, ticket.prefix_receipt, false, true,
                              &info)
            : authority.complete(ticket.prefix_token, false, true);
    clean = terminal && clean;
    if (terminal) {
      ticket.prefix_token = 0u;
      ticket.intermediate_dirty = false;
    } else {
      report(Status::fail(Reason::PipelineBusy), Check::Recover,
             ticket.prefix_epoch.ordinal, ticket.prefix_receipt.token(),
             ticket.prefix_receipt.generation(), cpu ? &info : nullptr);
    }
  }
  if (ticket.supply_receipt) {
    residency::CloseInfo info{};
    const std::uint64_t token = ticket.supply_receipt.token();
    const std::uint64_t generation = ticket.supply_receipt.generation();
    const bool terminal =
        close_cpu_epoch(authority, ticket.supply_receipt, false, true, &info);
    clean = terminal && clean;
    if (!terminal) {
      report(Status::fail(Reason::PipelineBusy), Check::Recover, Fail::NoEpoch,
             token, generation, &info);
    }
  }
  if (ticket.middle_receipt) {
    residency::CloseInfo info{};
    const std::uint64_t token = ticket.middle_receipt.token();
    const std::uint64_t generation = ticket.middle_receipt.generation();
    const bool terminal =
        close_cpu_epoch(authority, ticket.middle_receipt, false, true, &info);
    clean = terminal && clean;
    if (!terminal) {
      report(Status::fail(Reason::PipelineBusy), Check::Recover, Fail::NoEpoch,
             token, generation, &info);
    }
  }
  while (ticket.host_ready_count != 0u) {
    const std::size_t index = ticket.host_ready_count - 1u;
    std::destroy_at(&ticket.host_ready[index]);
    std::construct_at(&ticket.host_ready[index]);
    --ticket.host_ready_count;
  }
  if (ticket.host_ready_count == 0u) {
    ticket.forecast_stage = 0u;
    ticket.forecast_resources = {};
  }
  // A live GraphPersist owns the Host-output rows until its exact recovery or
  // terminal release. Do not consume those rows through the generic discard
  // path while the move-only credential is still armed.
  if (ticket.output_dirty && !ticket.output_persist) {
    const bool cpu = ticket.collective != nullptr &&
                     ticket.collective->device != nullptr &&
                     ticket.collective->device->backend == Backend::Cpu;
    if (!cpu || !persists.has_cpu_graph_retry(ticket.book_domain)) {
      const bool terminal = discard_keys(
          authority,
          std::span<const residency::CacheKey>{ticket.output_keys.data(),
                                               ticket.count},
          ticket.resident_output_region);
      clean = terminal && clean;
      ticket.output_dirty = !terminal;
    }
  }
  for (LiveResource &live : ticket.live_resources) {
    if (!live.dirty) {
      continue;
    }
    const bool terminal = discard_keys(
        authority,
        std::span<const residency::CacheKey>{live.keys.data(), live.count},
        live.region);
    clean = terminal && clean;
    if (terminal) {
      live = {};
    } else {
      report(Status::fail(Reason::PipelineBusy), Check::Recover, Fail::NoEpoch);
    }
  }
  ticket.intermediate_dirty = false;
  return clean;
}

} // namespace rund::compute::detail::graph_reduce

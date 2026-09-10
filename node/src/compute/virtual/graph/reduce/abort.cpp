#include "abort.hpp"

#include "cleanup.hpp"
#include "result.hpp"
#include "../../../device/residency/registry/graph_forecast_owner.hpp"
#include "../../../device/residency/registry/cpu_graph_owner.hpp"
#include "../../../device/residency/registry/graph_persist_owner.hpp"

#include <array>
#include <exception>

namespace rund::compute::detail::graph_reduce {

AbortController::AbortController(residency::Authority &authority,
                                 const std::span<Ticket> tickets,
                                 PrefetchController &prefetch,
                                 PersistController &persist,
                                 StageController &stages, Wavefront &wavefront,
                                 std::shared_ptr<CpuReceiptBook> receipts,
                                 std::shared_ptr<CpuGraphQuarantine> quarantine,
                                 FailLog &failure) noexcept
    : authority_(authority), tickets_(tickets), receipts_(std::move(receipts)),
      prefetch_(prefetch), persist_(persist), stages_(stages),
      wavefront_(wavefront), failure_(failure),
      quarantine_(std::move(quarantine)) {
  for (std::size_t index = 0u; index < tickets_.size(); ++index) {
    Ticket &ticket = tickets_[index];
    if (receipts_ != nullptr) {
      const bool prefix = receipts_->bind(ticket.prefix_receipt, authority_,
                                          index, CpuReceiptRole::Prefix) ==
                          CpuReceiptBindResult::Bound;
      const bool supply = receipts_->bind(ticket.supply_receipt, authority_,
                                          index, CpuReceiptRole::Supply) ==
                          CpuReceiptBindResult::Bound;
      const bool middle = receipts_->bind(ticket.middle_receipt, authority_,
                                          index, CpuReceiptRole::Middle) ==
                          CpuReceiptBindResult::Bound;
      const bool collective =
          receipts_->bind(ticket.collective_receipt, authority_, index,
                          CpuReceiptRole::Collective) ==
          CpuReceiptBindResult::Bound;
      receipts_bound_ =
          receipts_bound_ && prefix && supply && middle && collective;
    }
  }
}

VirtualGraphResult AbortController::finish(const Status status,
                                           const std::uint64_t page,
                                           bool child_poison) noexcept {
  if (!receipts_bound_) {
    failure_.note(Fail::NoStage, Fail::NoStage, Phase::Recovery,
                  Check::Authority, Status::fail(Reason::PipelineBusy));
    child_poison = true;
  }
  stages_.abort_active(child_poison);
  stages_.fold_pending(tickets_, child_poison);
  bool clean = prefetch_.cancel();
  clean = persist_.abort(tickets_, child_poison) && clean;
  const Fail primary = failure_.first();
  auto persist_owner = authority_.graph_persists();
  for (Ticket &ticket : tickets_) {
    child_poison = ticket.poison || child_poison;
    clean = cleanup_ticket(authority_, ticket, failure_) && clean;
  }
  // A Forecast whose first terminal/release raced an active alias remains in
  // its lane.  Retry that authenticated disposition only after all tickets
  // and persists have released their aliases, then require both workers to be
  // quiescent before returning the owner to its caller.
  clean = prefetch_.cancel() && clean;
  clean = prefetch_.quarantine() && clean;
  clean = authority_.graph_forecasts().recover_graph_forecast_quarantine() &&
          clean;
  clean = prefetch_.quiescent() && clean;
  for (Ticket &ticket : tickets_) {
    const bool cpu = ticket.collective != nullptr &&
                     ticket.collective->device != nullptr &&
                     ticket.collective->device->backend == Backend::Cpu;
    if (cpu && ticket.output_persist) {
      if (!ticket.output_persist.unknown()) {
        // The helper intentionally returns false after publishing the sticky
        // Unknown state; the fixed Unknown credential remains in Authority
        // until the Book snapshots it.
        static_cast<void>(
            persist_owner.recover_cpu_graph_persist(ticket.output_persist, true));
      }
    }
  }
  if (receipts_ != nullptr) {
    for (Ticket &ticket : tickets_) {
      const bool cpu = ticket.collective != nullptr &&
                       ticket.collective->device != nullptr &&
                       ticket.collective->device->backend == Backend::Cpu;
      if (!cpu || !ticket.output_persist) {
        continue;
      }
      if (!receipts_->hold_unknown(authority_, ticket.output_persist)) {
        std::terminate();
      }
    }
  } else {
    clean = false;
  }
  residency::CloseInfo close_info{};
  const bool recovered =
      receipts_ == nullptr || receipts_->recover(authority_, &close_info);
  if (!recovered) {
    if (close_info.check != residency::CloseInfo::Check::None) {
      if (primary.present) {
        failure_.attach(primary.stage, primary.batch, Phase::Recovery,
                        Check::Recover, primary.epoch, close_info);
      } else if (failure_.has()) {
        const Fail current = failure_.first();
        failure_.attach(current.stage, current.batch, Phase::Recovery,
                        Check::Recover, current.epoch, close_info);
      } else {
        failure_.note_cred(Fail::NoStage, Fail::NoStage, Phase::Recovery,
                           Check::Recover, Status::fail(Reason::PipelineBusy),
                           Fail::NoEpoch, close_info.token,
                           close_info.generation, &close_info);
      }
    } else if (!failure_.has()) {
      failure_.note(Fail::NoStage, Fail::NoStage, Phase::Recovery,
                    Check::Recover, Status::fail(Reason::PipelineBusy));
    }
  }
  clean = recovered && clean;
  if (receipts_ != nullptr && !receipts_->idle()) {
    const bool armed = quarantine_ != nullptr && receipts_ != nullptr &&
                       quarantine_->book == receipts_ &&
                       quarantine_->arm(quarantine_);
    const bool retained =
        armed && authority_.cpu_graph().retain_cpu_quarantine(quarantine_);
    if (!armed || !retained) {
      std::terminate();
    }
    if (retained) {
      quarantine_->commit();
      quarantine_->drop_handles();
    }
    clean = retained && clean;
  }
  wavefront_.abort();
  return failed(status, page, !clean || child_poison);
}

} // namespace rund::compute::detail::graph_reduce

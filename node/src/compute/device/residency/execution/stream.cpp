#include "stream.hpp"

#include "../registry/execution_owner.hpp"
#include <algorithm>

namespace rund::compute::detail::residency::execution {

ExecutionLease Stream::begin(Authority &authority, const Plan &plan) noexcept {
  if (authority_ != nullptr || plan.epoch_count() == 0u) {
    return ExecutionLease{.failure = AuthorityFailure::Invalid};
  }
  ExecutionLease lease = authority.executions().begin_execution_window(plan);
  if (!lease || !service_.bind(authority, plan, lease)) {
    return lease ? ExecutionLease{.failure = AuthorityFailure::Invalid} : lease;
  }
  authority_ = &authority;
  plan_ = &plan;
  lease_ = lease;
  return lease_;
}

ExecutionLease Stream::begin_schedule(Authority &authority,
                                      const Plan &plan) noexcept {
  const ExecutionLease lease = begin(authority, plan);
  if (lease) {
    scheduled_ = true;
  }
  return lease;
}

bool Stream::issue(const std::uint64_t epoch, const Phase phase,
                   ExecutionTicket &ticket) noexcept {
  ticket = {};
  if (authority_ == nullptr || plan_ == nullptr || closed_ || final_ ||
      epoch >= lease_.epochs ||
      (phase != Phase::Input && phase != Phase::Output)) {
    return false;
  }
  Slot &slot = slots_[epoch % WindowCapacity];
  if (slot.epoch != epoch) {
    if (slot.epoch != NeverUse &&
        (!slot.output_done || slot.epoch + WindowCapacity > epoch)) {
      return false;
    }
    slot = Slot{.epoch = epoch};
  }
  if ((phase == Phase::Input && slot.input_issued) ||
      (phase == Phase::Output &&
       (slot.output_issued || release_count_ <= epoch || !slot.release.status ||
        !slot.release.completed))) {
    return false;
  }
  if (!service_.issue(epoch, phase, ticket)) {
    return false;
  }
  (phase == Phase::Input ? slot.input_issued : slot.output_issued) = true;
  return true;
}

bool Stream::terminal(const ExecutionTicket &ticket,
                      const ExecutionTerminal terminal,
                      const Status status) noexcept {
  if (authority_ == nullptr || plan_ == nullptr || closed_ || final_ ||
      ticket.epoch >= lease_.epochs ||
      (ticket.phase != static_cast<std::uint8_t>(Phase::Input) &&
       ticket.phase != static_cast<std::uint8_t>(Phase::Output))) {
    return false;
  }
  Slot &slot = slots_[ticket.epoch % WindowCapacity];
  const bool input = ticket.phase == static_cast<std::uint8_t>(Phase::Input);
  if (slot.epoch != ticket.epoch ||
      (input ? (!slot.input_issued || slot.input_done)
             : (!slot.output_issued || slot.output_done)) ||
      !service_.terminal(ticket, terminal, status)) {
    return false;
  }
  (input ? slot.input_done : slot.output_done) = true;
  if (!status) {
    status_ = status;
    terminal_ = terminal == ExecutionTerminal::UnknownMayWrite
                    ? TerminalKind::UnknownMayWrite
                    : TerminalKind::Known;
  }
  return true;
}

bool Stream::release(const Release &release) noexcept {
  const std::uint64_t chunk_capacity =
      scheduled_ ? lease_.epochs
      : accepted_count_ >= lease_.epochs
          ? 0u
          : std::min<std::uint64_t>(WindowCapacity,
                                    lease_.epochs - accepted_count_);
  if (authority_ == nullptr || plan_ == nullptr || closed_ || final_ ||
      release.epoch != release_count_ ||
      release_count_ - accepted_count_ >= chunk_capacity) {
    return false;
  }
  Slot &slot = slots_[release.epoch % WindowCapacity];
  const bool suppressed =
      !release.status && release.terminal == TerminalKind::Known &&
      release.dispatched && release.completed && !release.may_write;
  const bool collapsed_unknown =
      terminal_ == TerminalKind::UnknownMayWrite && !release.status &&
      release.terminal == TerminalKind::UnknownMayWrite && release.dispatched &&
      !release.completed && release.may_write;
  const bool retired_suppressed =
      slot.epoch != NeverUse && slot.release.epoch == slot.epoch &&
      !slot.release.status && slot.release.terminal == TerminalKind::Known &&
      slot.release.dispatched && slot.release.completed &&
      !slot.release.may_write;
  if (slot.epoch != release.epoch &&
      (!(suppressed || collapsed_unknown) ||
       (slot.epoch != NeverUse && !slot.output_done &&
        !(scheduled_ && retired_suppressed)))) {
    return false;
  }
  if (slot.epoch != release.epoch) {
    slot = Slot{.epoch = release.epoch};
  }
  if ((release.dispatched && (release.status || release.may_write) &&
       !collapsed_unknown && !slot.input_done) ||
      !service_.release(release)) {
    return false;
  }
  slot.release = release;
  ++release_count_;
  if (!release.status) {
    status_ = release.status;
    terminal_ = release.terminal;
  }
  return true;
}

bool Stream::chunk(const WindowEvidence &evidence) noexcept {
  const std::size_t expected_count =
      accepted_count_ >= lease_.epochs
          ? 0u
          : static_cast<std::size_t>(std::min<std::uint64_t>(
                WindowCapacity, lease_.epochs - accepted_count_));
  if (authority_ == nullptr || plan_ == nullptr || scheduled_ || closed_ ||
      final_ || evidence.first_epoch != accepted_count_ ||
      evidence.release_count == 0u ||
      evidence.release_count != expected_count ||
      evidence.first_epoch + evidence.release_count != release_count_) {
    return false;
  }
  for (std::size_t index = 0u; index < evidence.release_count; ++index) {
    const std::uint64_t epoch = evidence.first_epoch + index;
    const Slot &slot = slots_[epoch % WindowCapacity];
    if (slot.epoch != epoch ||
        slot.release.backend_sequence !=
            evidence.releases[index].backend_sequence ||
        slot.release.epoch != evidence.releases[index].epoch) {
      return false;
    }
  }
  if (!authority_->executions().accept_execution_window(*plan_, evidence)) {
    return false;
  }
  ++chunk_count_;
  accepted_count_ += evidence.release_count;
  native_batches_ += evidence.native_batches;
  queue_calls_ += evidence.queue_calls;
  native_inflight_peak_ =
      std::max(native_inflight_peak_, evidence.native_inflight_peak);
  completed_ns_ = evidence.completed_ns;
  if (!evidence.status) {
    status_ = evidence.status;
    terminal_ = evidence.terminal;
  }
  final_ = !status_ || terminal_ == TerminalKind::UnknownMayWrite ||
           accepted_count_ == lease_.epochs;
  return true;
}

bool Stream::schedule(const ScheduleEvidence &evidence) noexcept {
  const bool unknown =
      evidence.terminal == TerminalKind::UnknownMayWrite && !evidence.status;
  const bool release_extent =
      unknown ? (release_count_ != 0u && release_count_ <= lease_.epochs)
              : release_count_ == lease_.epochs;
  if (authority_ == nullptr || plan_ == nullptr || !scheduled_ || closed_ ||
      final_ || !release_extent ||
      evidence.plan_identity != lease_.plan || evidence.token != lease_.token ||
      evidence.generation != lease_.generation ||
      evidence.epoch_count != lease_.epochs || evidence.public_handoffs != 1u ||
      evidence.native_batches != lease_.epochs || evidence.queue_calls == 0u ||
      evidence.queue_calls > evidence.native_batches ||
      evidence.native_inflight_peak == 0u ||
      evidence.native_inflight_peak > BankCapacity ||
      evidence.released_prefix != release_count_ ||
      evidence.completed_ns == 0u ||
      static_cast<bool>(evidence.status) != static_cast<bool>(status_) ||
      evidence.terminal != terminal_ ||
      !authority_->executions().accept_execution_schedule(*plan_, evidence)) {
    return false;
  }
  status_ = evidence.status;
  terminal_ = evidence.terminal;
  chunk_count_ = 1u;
  accepted_count_ = release_count_;
  native_batches_ = evidence.native_batches;
  queue_calls_ = evidence.queue_calls;
  native_inflight_peak_ = evidence.native_inflight_peak;
  completed_ns_ = evidence.completed_ns;
  final_ = true;
  return true;
}

bool Stream::abort(const Status failure,
                   const std::uint64_t completed_ns) noexcept {
  if (authority_ == nullptr || plan_ == nullptr || closed_ || final_ ||
      failure || completed_ns == 0u || accepted_count_ == 0u ||
      accepted_count_ != release_count_ || accepted_count_ >= lease_.epochs ||
      !(closed_result_ = authority_->executions().abort_execution_stream(
            lease_.token, lease_.generation, *plan_, failure))) {
    return false;
  }
  status_ = failure;
  terminal_ = TerminalKind::Known;
  completed_ns_ = completed_ns;
  final_ = true;
  closed_ = true;
  return true;
}

bool Stream::abandon_final(const WindowEvidence &evidence) noexcept {
  if (authority_ == nullptr || plan_ == nullptr || closed_ || final_ ||
      evidence.terminal != TerminalKind::Known ||
      evidence.first_epoch != accepted_count_ ||
      evidence.first_epoch + evidence.release_count != release_count_ ||
      !authority_->executions().abandon_execution_stream(*plan_, evidence)) {
    return false;
  }
  status_ = Status::fail(Reason::CompletionInvalid);
  terminal_ = TerminalKind::Known;
  closed_ = true;
  return true;
}

bool Stream::final(StreamEvidence &evidence) const noexcept {
  evidence = {};
  if (!final_ || completed_ns_ == 0u) {
    return false;
  }
  evidence = StreamEvidence{
      .status = status_,
      .terminal = terminal_,
      .plan_identity = lease_.plan,
      .token = lease_.token,
      .generation = lease_.generation,
      .epoch_count = lease_.epochs,
      .public_handoffs = 1u,
      .chunk_count = chunk_count_,
      .native_batches = native_batches_,
      .queue_calls = queue_calls_,
      .native_inflight_peak = native_inflight_peak_,
      .released_prefix = release_count_,
      .completed_ns = completed_ns_,
  };
  return true;
}

ExecutionClose Stream::close() noexcept {
  if (authority_ == nullptr || plan_ == nullptr || !final_) {
    return ExecutionClose{.failure = !final_ ? AuthorityFailure::Busy
                                             : AuthorityFailure::Invalid};
  }
  if (closed_) {
    return closed_result_;
  }
  ExecutionClose result =
      authority_->executions().close_execution(lease_.token, lease_.generation, *plan_);
  closed_ = static_cast<bool>(result);
  if (closed_) {
    closed_result_ = result;
  }
  return result;
}

bool Stream::abandon() noexcept {
  if (authority_ == nullptr || plan_ == nullptr || closed_ || final_ ||
      release_count_ != 0u) {
    return false;
  }
  const bool abandoned =
      authority_->executions().abandon_execution(lease_.token, lease_.generation, *plan_);
  closed_ = abandoned;
  return abandoned;
}

} // namespace rund::compute::detail::residency::execution

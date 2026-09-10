#include "../window.hpp"

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>

namespace rund::compute::detail {

bool PipelineExecutionWindow::bind(
    const residency::execution::Plan &plan,
    const residency::ExecutionLease &lease) noexcept {
  if (plan.epoch_count() > residency::execution::WindowCapacity) {
    return false;
  }
  return bind(plan, lease, 0u, static_cast<std::size_t>(plan.epoch_count()));
}

bool PipelineExecutionWindow::bind(const residency::execution::Plan &plan,
                                   const residency::ExecutionLease &lease,
                                   const std::uint64_t first_epoch,
                                   const std::size_t count) noexcept {
  if (plan_ != nullptr || !lease || plan.identity() == 0u ||
      plan.epoch_count() == 0u || count == 0u ||
      count > residency::execution::WindowCapacity ||
      first_epoch >= plan.epoch_count() ||
      count > plan.epoch_count() - first_epoch ||
      lease.plan != plan.identity() || lease.epochs != plan.epoch_count()) {
    return false;
  }
  plan_ = &plan;
  lease_ = lease;
  first_epoch_ = first_epoch;
  expected_count_ = count;
  for (std::size_t slot = 0u; slot < count; ++slot) {
    slot_epoch_[slot] = first_epoch + slot;
  }
  return true;
}

void PipelineExecutionWindow::reset() noexcept { *this = {}; }

bool PipelineExecutionWindow::terminal(
    const PipelineWindowRelease &value) noexcept {
  const residency::execution::Release &release = value.release;
  const bool suppressed = !release.status && release.dispatched &&
                          release.completed && !release.may_write;
  const bool collapsed_unknown =
      !release.status && release.dispatched && !release.completed &&
      release.may_write &&
      release.terminal == residency::execution::TerminalKind::UnknownMayWrite;
  const bool canonical =
      value.terminal_published && value.reseeded &&
      value.published_generation ==
          value.attempt_generation + (release.status ? 1u : 0u);
  if (plan_ == nullptr || count_ >= expected_count_ ||
      release.plan_identity != lease_.plan || release.token != lease_.token ||
      release.generation != lease_.generation ||
      release.epoch != slot_epoch_[count_] ||
      release.backend_sequence != release.epoch + 1u ||
      release.bank != release.epoch % residency::execution::BankCapacity ||
      (!canonical &&
       !((suppressed || collapsed_unknown) && !value.terminal_published &&
         !value.reseeded && value.attempt_generation == 0u &&
         value.published_generation == 0u)) ||
      (release.dispatched && count_ != 0u &&
       !releases_[count_ - 1u].release.dispatched)) {
    return false;
  }
  releases_[count_++] = value;
  failed_ = failed_ || !release.status;
  unknown_ =
      unknown_ ||
      release.terminal == residency::execution::TerminalKind::UnknownMayWrite;
  return true;
}

bool PipelineExecutionWindow::final(
    const std::uint64_t queue_calls, const std::uint64_t native_inflight_peak,
    const std::uint64_t completed_ns,
    residency::execution::WindowEvidence &evidence) noexcept {
  evidence = {};
  if (plan_ == nullptr || count_ != expected_count_ || completed_ns == 0u) {
    return false;
  }
  const residency::execution::Release *failure = nullptr;
  bool unknown = false;
  std::uint64_t native_batches = 0u;
  std::array<residency::execution::Release,
             residency::execution::WindowCapacity>
      projected{};
  for (std::size_t index = 0u; index < count_; ++index) {
    projected[index] = releases_[index].release;
    native_batches += releases_[index].release.dispatched ? 1u : 0u;
    if (failure == nullptr && !releases_[index].release.status) {
      failure = &releases_[index].release;
    }
    unknown =
        unknown || releases_[index].release.terminal ==
                       residency::execution::TerminalKind::UnknownMayWrite;
  }
  if ((native_batches != 0u &&
       (queue_calls == 0u || native_inflight_peak == 0u ||
        native_inflight_peak > residency::execution::BankCapacity)) ||
      (native_batches == 0u && native_inflight_peak != 0u)) {
    return false;
  }
  evidence = residency::execution::WindowEvidence{
      .status = failure == nullptr ? Status::success() : failure->status,
      .terminal = unknown ? residency::execution::TerminalKind::UnknownMayWrite
                  : failure == nullptr
                      ? residency::execution::TerminalKind::Known
                      : failure->terminal,
      .plan_identity = lease_.plan,
      .token = lease_.token,
      .generation = lease_.generation,
      .first_epoch = first_epoch_,
      .epoch_count = lease_.epochs,
      .public_handoffs = 1u,
      .native_batches = native_batches,
      .queue_calls = queue_calls,
      .native_inflight_peak = native_inflight_peak,
      .releases = projected,
      .release_count = count_,
      .completed_ns = completed_ns,
  };
  return true;
}

bool PipelineExecutionWindow::integrity(
    const std::uint64_t queue_calls, const std::uint64_t native_inflight_peak,
    const std::uint64_t completed_ns,
    residency::execution::WindowEvidence &evidence) const noexcept {
  evidence = {};
  if (plan_ == nullptr || count_ != expected_count_ || count_ == 0u) {
    return false;
  }
  std::uint64_t native_batches = 0u;
  bool unknown = false;
  for (std::size_t index = 0u; index < count_; ++index) {
    evidence.releases[index] = releases_[index].release;
    native_batches += releases_[index].release.dispatched ? 1u : 0u;
    unknown =
        unknown || releases_[index].release.terminal ==
                       residency::execution::TerminalKind::UnknownMayWrite;
  }
  evidence.status = Status::fail(Reason::CompletionInvalid);
  evidence.terminal = unknown
                          ? residency::execution::TerminalKind::UnknownMayWrite
                          : residency::execution::TerminalKind::Known;
  evidence.plan_identity = lease_.plan;
  evidence.token = lease_.token;
  evidence.generation = lease_.generation;
  evidence.first_epoch = first_epoch_;
  evidence.epoch_count = lease_.epochs;
  evidence.public_handoffs = 1u;
  evidence.native_batches = native_batches;
  evidence.queue_calls = queue_calls;
  evidence.native_inflight_peak = native_inflight_peak;
  evidence.release_count = count_;
  evidence.completed_ns = completed_ns;
  return true;
}

} // namespace rund::compute::detail

#include "receipt.hpp"

namespace rund::compute::detail::residency::execution {

bool Receipt::arm(const Plan &plan, const std::uint64_t token,
                  const std::uint64_t generation,
                  const std::uint64_t started_ns) noexcept {
  if (plan.identity() == 0u || plan.epoch_count() == 0u || token == 0u ||
      generation == 0u || started_ns == 0u) {
    return false;
  }
  std::lock_guard lock{gate_};
  if (state_ != State::Idle) {
    return false;
  }
  snapshot_ = ReceiptSnapshot{.status = Status::success(),
                              .plan_identity = plan.identity(),
                              .token = token,
                              .generation = generation,
                              .epoch_count = plan.epoch_count(),
                              .started_ns = started_ns};
  state_ = State::Armed;
  return true;
}

bool Receipt::submit() noexcept {
  std::lock_guard lock{gate_};
  if (state_ != State::Armed) {
    return false;
  }
  state_ = State::Submitted;
  return true;
}

bool Receipt::terminal(const Evidence &terminal) noexcept {
  std::lock_guard lock{gate_};
  if (state_ != State::Submitted ||
      terminal.plan_identity != snapshot_.plan_identity ||
      terminal.token != snapshot_.token ||
      terminal.generation != snapshot_.generation ||
      terminal.epoch_count != snapshot_.epoch_count ||
      terminal.native_submissions != 1u ||
      terminal.failure_count > terminal.failures.size() ||
      terminal.completed_ns < snapshot_.started_ns ||
      (terminal.terminal == TerminalKind::UnknownMayWrite && terminal.status) ||
      (terminal.status && terminal.terminal != TerminalKind::Known) ||
      (terminal.status &&
       (terminal.progress.input_services != snapshot_.epoch_count ||
        terminal.progress.native_dispatches != snapshot_.epoch_count ||
        terminal.progress.native_completions != snapshot_.epoch_count ||
        terminal.progress.output_services != snapshot_.epoch_count ||
        terminal.progress.native_inflight_peak == 0u ||
        terminal.progress.native_inflight_peak > BankCapacity))) {
    return false;
  }
  snapshot_.status = terminal.status;
  snapshot_.terminal = terminal.terminal;
  snapshot_.native_submissions = terminal.native_submissions;
  snapshot_.progress = terminal.progress;
  snapshot_.failures = terminal.failures;
  snapshot_.failure_count = terminal.failure_count;
  snapshot_.completed_ns = terminal.completed_ns;
  state_ = State::Ready;
  ready_.notify_one();
  return true;
}

ReceiptSnapshot Receipt::wait() noexcept {
  std::unique_lock lock{gate_};
  ready_.wait(lock, [this] { return state_ == State::Ready; });
  const ReceiptSnapshot result = snapshot_;
  snapshot_ = {};
  state_ = State::Idle;
  return result;
}

} // namespace rund::compute::detail::residency::execution

#include "../../registry/execution_owner.hpp"
#include "../run.hpp"

#include <algorithm>

namespace rund::compute::detail::residency::execution {

bool Run::merge(const FailureEvidence value) noexcept {
  if (value.phases == 0u) {
    return false;
  }
  for (std::size_t index = 0u; index < host_failure_count_; ++index) {
    FailureEvidence &prior = host_failures_[index];
    if (prior.epoch == value.epoch) {
      prior.phases = static_cast<std::uint8_t>(prior.phases | value.phases);
      prior.may_write =
          static_cast<std::uint8_t>(prior.may_write | value.may_write);
      prior.terminal =
          static_cast<std::uint8_t>(prior.terminal | value.terminal);
      return true;
    }
  }
  if (host_failure_count_ == host_failures_.size()) {
    return false;
  }
  host_failures_[host_failure_count_++] = value;
  return true;
}

ExecutionClose Run::close(const std::uint64_t completed_ns) noexcept {
  if (authority_ == nullptr || plan_ == nullptr || closed_) {
    return ExecutionClose{.failure = AuthorityFailure::Invalid};
  }
  if (!native_done_) {
    if (native_rejected_) {
      ExecutionClose result =
          authority_->executions().close_execution(lease_.token, lease_.generation, *plan_);
      closed_ = static_cast<bool>(result);
      return result;
    }
    if (host_failure_count_ != 0u) {
      ExecutionClose result =
          authority_->executions().close_execution(lease_.token, lease_.generation, *plan_);
      closed_ = static_cast<bool>(result);
      return result;
    }
    return ExecutionClose{.failure = AuthorityFailure::Busy};
  }

  std::array<FailureEvidence, FailureCapacity> combined = host_failures_;
  const std::size_t host_count = host_failure_count_;
  for (std::size_t index = 0u; index < native_.failure_count; ++index) {
    if (!merge(native_.failures[index])) {
      host_failures_ = combined;
      host_failure_count_ = host_count;
      return ExecutionClose{.failure = AuthorityFailure::Capacity};
    }
  }
  std::sort(host_failures_.begin(),
            host_failures_.begin() +
                static_cast<std::ptrdiff_t>(host_failure_count_),
            [](const FailureEvidence left, const FailureEvidence right) {
              return left.epoch < right.epoch;
            });
  const bool successful =
      static_cast<bool>(native_.status) && host_failure_count_ == 0u;
  const TerminalKind terminal =
      host_unknown_ || native_.terminal == TerminalKind::UnknownMayWrite
          ? TerminalKind::UnknownMayWrite
          : TerminalKind::Known;
  Evidence evidence{
      .status = successful ? Status::success()
                           : (host_count != 0u ? host_status_ : native_.status),
      .terminal = terminal,
      .plan_identity = native_.plan_identity,
      .token = native_.token,
      .generation = native_.generation,
      .epoch_count = native_.epoch_count,
      .native_submissions = native_.native_submissions,
      .progress =
          Progress{
              .input_services = input_done_ ? 1u : 0u,
              .native_dispatches = native_.native_dispatches,
              .native_completions = native_.native_completions,
              .output_services = output_done_ ? 1u : 0u,
              .native_inflight_peak = native_.native_inflight_peak,
          },
      .failure_count = host_failure_count_,
      .completed_ns = completed_ns,
  };
  std::copy_n(host_failures_.begin(), host_failure_count_,
              evidence.failures.begin());
  ExecutionClose result = authority_->executions().close_execution(*plan_, evidence);
  if (result) {
    closed_ = true;
  } else {
    host_failures_ = combined;
    host_failure_count_ = host_count;
  }
  return result;
}

} // namespace rund::compute::detail::residency::execution

#include "internal.hpp"

namespace rund::compute::detail::residency::execution::authority_close_detail {

Validation validate(const registry_model::ExecutionSlot &slot, const Plan &plan,
                    const Evidence &evidence,
                    const bool authority_ready) noexcept {
  Validation result{};
  if (!authority_ready) {
    result.failure = AuthorityFailure::Invalid;
    return result;
  }
  const bool coordinated = slot.native_accepted;
  if (slot.token == 0u || evidence.token != slot.token ||
      slot.sliding_admitted || evidence.generation != slot.generation ||
      evidence.plan_identity != slot.plan || evidence.epoch_count != slot.epochs ||
      plan.identity() != slot.plan || plan.epoch_count() != slot.epochs ||
      evidence.native_submissions != 1u ||
      (evidence.terminal == TerminalKind::UnknownMayWrite && evidence.status) ||
      (evidence.status && evidence.terminal != TerminalKind::Known) ||
      evidence.failure_count > evidence.failures.size() ||
      (slot.cache_admitted && !coordinated) ||
      (!coordinated &&
       (slot.next_sequence != 1u || slot.progress.input_services != 0u ||
        slot.progress.native_dispatches != 0u ||
        slot.progress.native_completions != 0u ||
        slot.progress.output_services != 0u || slot.native_inflight != 0u))) {
    result.failure = AuthorityFailure::Invalid;
    return result;
  }
  result.progress = ExecutionProgress{
      .input_services = evidence.progress.input_services,
      .native_dispatches = evidence.progress.native_dispatches,
      .native_completions = evidence.progress.native_completions,
      .output_services = evidence.progress.output_services,
      .native_inflight_peak = evidence.progress.native_inflight_peak,
  };
  if (result.progress.input_services > slot.epochs ||
      result.progress.native_dispatches > slot.epochs ||
      result.progress.native_completions > result.progress.native_dispatches ||
      result.progress.output_services > result.progress.native_completions ||
      result.progress.native_inflight_peak > BankCapacity ||
      (result.progress.native_dispatches != 0u &&
       result.progress.native_inflight_peak == 0u)) {
    result.failure = AuthorityFailure::Invalid;
    return result;
  }

  if (coordinated) {
    const NativeEvidence &native = slot.native;
    if (slot.epochs != 1u ||
        evidence.native_submissions != native.native_submissions ||
        result.progress.input_services != slot.progress.input_services ||
        result.progress.native_dispatches != native.native_dispatches ||
        result.progress.native_completions != native.native_completions ||
        result.progress.output_services != slot.progress.output_services ||
        result.progress.native_inflight_peak != native.native_inflight_peak ||
        evidence.completed_ns < native.completed_ns) {
      result.failure = AuthorityFailure::Invalid;
      return result;
    }
    if (native.status &&
        (slot.progress.output_services != slot.epochs ||
         slot.native_inflight != 0u)) {
      result.failure = AuthorityFailure::Busy;
      return result;
    }
    if (slot.failed && !slot.unknown) {
      for (std::size_t bank = 0u; bank < BankCapacity; ++bank) {
        for (std::size_t phase = 0u; phase < 3u; ++phase) {
          if (slot.issued[bank][phase] != NeverUse &&
              slot.may_write[bank][phase] &&
              slot.terminals[bank][phase] != slot.issued[bank][phase]) {
            result.failure = AuthorityFailure::Busy;
            return result;
          }
        }
      }
    }

    FailureEvidence expected{};
    bool has_expected = false;
    const auto merge_expected =
        [&expected, &has_expected](const FailureEvidence value) {
          if (!has_expected) {
            expected = value;
            has_expected = true;
            return;
          }
          expected.phases =
              static_cast<std::uint8_t>(expected.phases | value.phases);
          expected.may_write =
              static_cast<std::uint8_t>(expected.may_write | value.may_write);
          expected.terminal =
              static_cast<std::uint8_t>(expected.terminal | value.terminal);
        };
    for (std::size_t index = 0u; index < native.failure_count; ++index) {
      merge_expected(native.failures[index]);
    }
    for (std::size_t index = 0u; index < slot.failure_count; ++index) {
      const ExecutionFailure failure = slot.failures[index];
      merge_expected(FailureEvidence{.epoch = failure.epoch,
                                     .phases = failure.phases,
                                     .may_write = failure.may_write,
                                     .terminal = failure.terminal});
    }
    const bool expected_success = !slot.failed;
    const TerminalKind expected_terminal =
        slot.unknown ? TerminalKind::UnknownMayWrite : TerminalKind::Known;
    if (static_cast<bool>(evidence.status) != expected_success ||
        evidence.terminal != expected_terminal ||
        evidence.failure_count != (has_expected ? 1u : 0u) ||
        (has_expected &&
         (evidence.failures[0].epoch != expected.epoch ||
          evidence.failures[0].phases != expected.phases ||
          evidence.failures[0].may_write != expected.may_write ||
          evidence.failures[0].terminal != expected.terminal))) {
      result.failure = AuthorityFailure::Invalid;
      return result;
    }
  }
  result.successful = static_cast<bool>(evidence.status);
  if (result.successful &&
      (evidence.failure_count != 0u ||
       result.progress.input_services != slot.epochs ||
       result.progress.native_dispatches != slot.epochs ||
       result.progress.native_completions != slot.epochs ||
       result.progress.output_services != slot.epochs)) {
    result.failure = AuthorityFailure::Invalid;
    return result;
  }
  if (!result.successful && evidence.failure_count == 0u) {
    result.failure = AuthorityFailure::Invalid;
    return result;
  }
  for (std::size_t index = 0u; index < evidence.failure_count; ++index) {
    const FailureEvidence failure = evidence.failures[index];
    if (failure.epoch >= slot.epochs || failure.phases == 0u ||
        (failure.phases & ~std::uint8_t{7u}) != 0u ||
        (failure.may_write & ~std::uint8_t{7u}) != 0u ||
        (failure.terminal & ~std::uint8_t{7u}) != 0u ||
        (failure.phases & ~failure.may_write) != 0u ||
        (evidence.terminal == TerminalKind::Known &&
         (failure.may_write & ~failure.terminal) != 0u)) {
      result.failure = AuthorityFailure::Invalid;
      return result;
    }
    for (std::size_t prior = 0u; prior < index; ++prior) {
      if (evidence.failures[prior].epoch >= failure.epoch) {
        result.failure = AuthorityFailure::Invalid;
        return result;
      }
    }
  }
  return result;
}

} // namespace rund::compute::detail::residency::execution::authority_close_detail

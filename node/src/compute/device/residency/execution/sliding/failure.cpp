#include "internal.hpp"

namespace rund::compute::detail::residency::execution {

void Sliding::State::record_failure(const SlidingCoordinate coordinate,
                                    const Status failure,
                                    const TerminalKind certainty,
                                    const bool may_write) noexcept {
  const bool earlier =
      !has_failure || coordinate.ordinal < first_failure.ordinal;
  if (earlier) {
    first_failure = coordinate;
    status = failure ? integrity_failure() : failure;
    first_failure_terminal = certainty;
    first_failure_may_write = may_write;
    has_failure = true;
  } else if (coordinate == first_failure) {
    first_failure_may_write = first_failure_may_write || may_write;
    if (certainty == TerminalKind::UnknownMayWrite) {
      first_failure_terminal = certainty;
    }
  }
  if (certainty == TerminalKind::UnknownMayWrite) {
    terminal = certainty;
    quarantine = true;
  }
}

void Sliding::State::mark_unknown(const Status failure) noexcept {
  if (!has_failure) {
    first_failure = {};
    status = failure ? integrity_failure() : failure;
    first_failure_terminal = TerminalKind::UnknownMayWrite;
    first_failure_may_write = true;
    has_failure = true;
  }
  terminal = TerminalKind::UnknownMayWrite;
  quarantine = true;
}

void Sliding::State::cancel_input_suffix(
    const std::uint64_t boundary) noexcept {
  for (InputCell &cell : inputs) {
    if (cell.state == InputState::Ready &&
        cell.coordinate.ordinal >= boundary && !cell.physical_handoff) {
      cell.state = InputState::Free;
    }
  }
}

void Sliding::State::fail_native_suffix(const std::uint64_t boundary) noexcept {
  for (NativeCell &cell : native_cells) {
    if (cell.coordinate.ordinal < boundary) {
      continue;
    }
    if (cell.state == NativeState::Submitted) {
      cell.suppress = true;
      continue;
    }
    if (cell.state == NativeState::Ready) {
      cell.state = NativeState::Invalidating;
      for (OutputCell &output : outputs) {
        if (output.coordinate == cell.coordinate &&
            output.state == OutputState::Reserved) {
          output.state = OutputState::Free;
        }
      }
      continue;
    }
    if (cell.state != NativeState::Draining) {
      continue;
    }

    cell.invalidate_after_terminal = true;
    const bool output_callback_live = std::any_of(
        outputs.begin(), outputs.end(), [&](const OutputCell &output) {
          return output.coordinate == cell.coordinate &&
                 (output.physical_handoff ||
                  output.state == OutputState::Draining ||
                  output.state == OutputState::Completing ||
                  output.state == OutputState::Persisting);
        });
    if (model_only || cell.physical_handoff || output_callback_live) {
      continue;
    }

    // A bound run aborts the whole reserved physical generation once. A
    // suffix whose Native callback returned has no reason to issue D2H or
    // Persist merely to manufacture cell terminals.
    for (OutputCell &output : outputs) {
      if (output.coordinate == cell.coordinate &&
          (output.state == OutputState::Reserved ||
           output.state == OutputState::Ready ||
           output.state == OutputState::Invalidating)) {
        output.state = OutputState::Free;
        output.physical_handoff = false;
      }
    }
    TerminalCell &terminal_cell = terminals[static_cast<std::size_t>(
        cell.coordinate.ordinal % SlidingNativeCapacity)];
    terminal_cell = TerminalCell{.ordinal = cell.coordinate.ordinal,
                                 .terminal = true,
                                 .may_write = true,
                                 .invalidating = false};
    cell.state = NativeState::Free;
    cell.resolved = cell.output_count;
  }
}

void Sliding::State::fail_output_suffix(const std::uint64_t boundary) noexcept {
  for (OutputCell &cell : outputs) {
    if (cell.coordinate.ordinal < boundary) {
      continue;
    }
    if (cell.state == OutputState::Ready) {
      cell.state = OutputState::Invalidating;
    } else if (cell.state == OutputState::Persisted) {
      ++persist_completed_after_failure;
      static_cast<void>(add(persist_bytes_after_failure, cell.expected_bytes));
      cell.state = OutputState::Free;
    } else if (cell.state == OutputState::Persisting ||
               cell.state == OutputState::Draining ||
               cell.state == OutputState::Completing) {
      cell.invalidate_after_terminal = true;
    }
  }
}

void Sliding::State::fail_terminal_suffix(
    const std::uint64_t boundary) noexcept {
  for (TerminalCell &cell : terminals) {
    if (cell.ordinal != NeverUse && cell.ordinal >= boundary && cell.terminal &&
        cell.may_write) {
      cell.terminal = false;
      cell.invalidating = true;
    }
  }
}

void Sliding::State::fail(const SlidingCoordinate coordinate,
                          const Status failure, const TerminalKind certainty,
                          const bool may_write) noexcept {
  record_failure(coordinate, failure, certainty, may_write);
  const std::uint64_t boundary = first_failure.ordinal;
  cancel_input_suffix(boundary);
  fail_native_suffix(boundary);
  fail_output_suffix(boundary);
  fail_terminal_suffix(boundary);
}

} // namespace rund::compute::detail::residency::execution

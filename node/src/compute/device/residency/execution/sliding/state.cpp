#include "internal.hpp"

#include <algorithm>
#include <limits>
#include <utility>

namespace rund::compute::detail::residency::execution {
namespace {

[[nodiscard]] bool same_projection(const SlidingProjection &left,
                                   const SlidingProjection &right) noexcept {
  return left.coordinate == right.coordinate &&
         left.prefetch_epoch == right.prefetch_epoch &&
         left.ready_epoch == right.ready_epoch &&
         left.fetch_bytes == right.fetch_bytes &&
         left.persist_bytes == right.persist_bytes &&
         left.use_count == right.use_count &&
         left.fetch_count == right.fetch_count &&
         left.persist_count == right.persist_count;
}

} // namespace

[[nodiscard]] bool add(std::uint64_t &value,
                       const std::uint64_t increment) noexcept {
  if (increment > std::numeric_limits<std::uint64_t>::max() - value) {
    return false;
  }
  value += increment;
  return true;
}

[[nodiscard]] Status integrity_failure() noexcept {
  return Status::fail(Reason::CompletionInvalid);
}

Sliding::State::State(
    SlidingInvocation value, const std::uint64_t authority_token,
    const std::uint64_t run_generation, const std::uint64_t owner_nonce,
    const std::uint32_t input_capacity, const std::uint32_t output_capacity,
    const std::size_t required_inputs, const std::size_t required_outputs,
    const bool model) noexcept
    : invocation(std::move(value)), plan(invocation.identity()),
      token(authority_token), generation(run_generation), owner(owner_nonce),
      planned(invocation.count()),
      host_bank_count(invocation.topology() == SlidingTopology::Direct
                          ? static_cast<std::uint32_t>(BankCapacity)
                          : 1u),
      input_stride(input_capacity), output_stride(output_capacity),
      input_count(input_capacity * host_bank_count),
      output_count(output_capacity * host_bank_count),
      coordinate_input_count(required_inputs),
      coordinate_output_count(required_outputs), model_only(model) {}

[[nodiscard]] bool Sliding::State::runnable() const noexcept {
  return !closed && !finalizing && (model_only || authority_bound);
}

[[nodiscard]] bool
Sliding::State::credential(const SlidingTicket &ticket) const noexcept {
  return ticket.plan == plan && ticket.token == token &&
         ticket.generation == generation && ticket.owner == owner;
}

[[nodiscard]] bool
Sliding::State::exact(const SlidingProjection &supplied,
                      const std::span<residency::PageUse> scratch,
                      SlidingProjection &expected) const noexcept {
  return invocation.project(supplied.coordinate.ordinal, scratch, expected) &&
         same_projection(supplied, expected);
}

[[nodiscard]] bool
Sliding::State::accepts(const SlidingCoordinate coordinate) const noexcept {
  return !has_failure || coordinate.ordinal < first_failure.ordinal;
}

[[nodiscard]] Sliding::State::InputCell *
Sliding::State::input(const SlidingTicket &ticket,
                      const SlidingTicketKind kind) noexcept {
  if (!credential(ticket) || ticket.kind != kind ||
      ticket.slot >= input_stride) {
    return nullptr;
  }
  InputCell &cell = inputs[input_offset(ticket.coordinate) + ticket.slot];
  return cell.turn == ticket.turn && cell.coordinate == ticket.coordinate &&
                 cell.use == ticket.use &&
                 cell.expected_bytes == ticket.expected_bytes
             ? &cell
             : nullptr;
}

[[nodiscard]] Sliding::State::OutputCell *
Sliding::State::output(const SlidingTicket &ticket,
                       const SlidingTicketKind kind) noexcept {
  if (!credential(ticket) || ticket.kind != kind ||
      ticket.slot >= output_stride) {
    return nullptr;
  }
  OutputCell &cell = outputs[output_offset(ticket.coordinate) + ticket.slot];
  return cell.turn == ticket.turn && cell.coordinate == ticket.coordinate &&
                 cell.use == ticket.use &&
                 cell.expected_bytes == ticket.expected_bytes
             ? &cell
             : nullptr;
}

[[nodiscard]] Sliding::State::NativeCell *
Sliding::State::native(const SlidingTicket &ticket,
                       const SlidingTicketKind kind) noexcept {
  if (!credential(ticket) || ticket.kind != kind ||
      ticket.slot >= SlidingNativeCapacity) {
    return nullptr;
  }
  NativeCell &cell = native_cells[ticket.slot];
  return cell.turn == ticket.turn && cell.coordinate == ticket.coordinate &&
                 cell.promote_bytes == ticket.expected_bytes
             ? &cell
             : nullptr;
}

[[nodiscard]] std::size_t
Sliding::State::host_bank(const SlidingCoordinate coordinate) const noexcept {
  return host_bank_count == 1u
             ? 0u
             : static_cast<std::size_t>(coordinate.ordinal % host_bank_count);
}

[[nodiscard]] std::size_t Sliding::State::input_offset(
    const SlidingCoordinate coordinate) const noexcept {
  return host_bank(coordinate) * input_stride;
}

[[nodiscard]] std::size_t Sliding::State::output_offset(
    const SlidingCoordinate coordinate) const noexcept {
  return host_bank(coordinate) * output_stride;
}

[[nodiscard]] std::uint64_t Sliding::State::bank_frontier(
    const SlidingCoordinate coordinate) const noexcept {
  if (host_bank_count == 1u ||
      host_bank(SlidingCoordinate{.ordinal = admitted}) ==
          host_bank(coordinate)) {
    return admitted;
  }
  // A supplied coordinate in the other bank proves admitted+1 exists and
  // has that bank, so the increment cannot overflow.
  return admitted + 1u;
}

[[nodiscard]] std::span<Sliding::State::InputCell>
Sliding::State::input_ring(const SlidingCoordinate coordinate) noexcept {
  return {inputs.data() + input_offset(coordinate), input_stride};
}

[[nodiscard]] std::span<Sliding::State::OutputCell>
Sliding::State::output_ring(const SlidingCoordinate coordinate) noexcept {
  return {outputs.data() + output_offset(coordinate), output_stride};
}

void Sliding::State::advance_frontier() noexcept {
  while (terminal_frontier < admitted) {
    TerminalCell &cell = terminals[static_cast<std::size_t>(
        terminal_frontier % SlidingNativeCapacity)];
    if (!cell.terminal || cell.invalidating ||
        cell.ordinal != terminal_frontier) {
      break;
    }
    cell = {};
    ++terminal_frontier;
  }
}

[[nodiscard]] bool Sliding::State::terminal_for(
    const SlidingCoordinate coordinate) const noexcept {
  if (coordinate.ordinal < terminal_frontier) {
    return true;
  }
  const TerminalCell &cell = terminals[static_cast<std::size_t>(
      coordinate.ordinal % SlidingNativeCapacity)];
  return cell.ordinal == coordinate.ordinal && cell.terminal &&
         !cell.invalidating;
}

void Sliding::State::advance_persist_frontier() noexcept {
  while (persist_frontier < persist_issued) {
    const auto found =
        std::find_if(outputs.begin(), outputs.begin() + output_count,
                     [&](const OutputCell &cell) {
                       return cell.state == OutputState::Persisted &&
                              cell.persist_sequence == persist_frontier;
                     });
    if (found == outputs.begin() + output_count) {
      break;
    }
    found->state = OutputState::Free;
    ++persist_frontier;
  }
}

void Sliding::State::retire(const SlidingCoordinate coordinate,
                            const bool may_write) noexcept {
  TerminalCell &cell = terminals[static_cast<std::size_t>(
      coordinate.ordinal % SlidingNativeCapacity)];
  cell = TerminalCell{
      .ordinal = coordinate.ordinal, .terminal = true, .may_write = may_write};
  if (has_failure && coordinate.ordinal >= first_failure.ordinal && may_write) {
    cell.terminal = false;
    cell.invalidating = true;
  }
  advance_frontier();
}

void Sliding::State::cancel_reserved(const SlidingCoordinate coordinate,
                                     NativeCell &native) noexcept {
  for (OutputCell &cell : outputs) {
    if (cell.coordinate == coordinate && cell.state == OutputState::Reserved) {
      cell.state = OutputState::Free;
      ++native.resolved;
    }
  }
}

void Sliding::State::resolve_native(NativeCell &native) noexcept {
  if (native.state == NativeState::Draining &&
      native.resolved == native.output_count) {
    const bool wrote = native.output_count != 0u;
    native.state = NativeState::Free;
    retire(native.coordinate, wrote);
  }
}

[[nodiscard]] bool Sliding::State::no_issued() const noexcept {
  return std::none_of(inputs.begin(), inputs.begin() + input_count,
                      [](const InputCell &cell) {
                        return cell.state == InputState::Fetching ||
                               cell.state == InputState::Completing ||
                               cell.state == InputState::Promoting ||
                               cell.state == InputState::Invalidating ||
                               cell.physical_handoff;
                      }) &&
         std::none_of(outputs.begin(), outputs.begin() + output_count,
                      [](const OutputCell &cell) {
                        return cell.state == OutputState::Reserved ||
                               cell.state == OutputState::Draining ||
                               cell.state == OutputState::Completing ||
                               cell.state == OutputState::Ready ||
                               cell.state == OutputState::Persisting ||
                               cell.state == OutputState::Persisted ||
                               cell.state == OutputState::Invalidating;
                      }) &&
         std::none_of(native_cells.begin(), native_cells.end(),
                      [](const NativeCell &cell) {
                        return cell.state == NativeState::Promoting ||
                               cell.state == NativeState::Completing ||
                               cell.state == NativeState::Ready ||
                               cell.state == NativeState::Submitted ||
                               cell.state == NativeState::Draining ||
                               cell.state == NativeState::Invalidating;
                      }) &&
         std::none_of(
             terminals.begin(), terminals.end(),
             [](const TerminalCell &cell) { return cell.invalidating; });
}

[[nodiscard]] bool Sliding::State::callbacks_quiesced() const noexcept {
  return std::none_of(inputs.begin(), inputs.begin() + input_count,
                      [](const InputCell &cell) {
                        return cell.state == InputState::Fetching ||
                               cell.state == InputState::Completing ||
                               cell.state == InputState::Promoting ||
                               cell.physical_handoff;
                      }) &&
         std::none_of(outputs.begin(), outputs.begin() + output_count,
                      [](const OutputCell &cell) {
                        return cell.state == OutputState::Draining ||
                               cell.state == OutputState::Completing ||
                               cell.state == OutputState::Persisting ||
                               cell.physical_handoff;
                      }) &&
         std::none_of(native_cells.begin(), native_cells.end(),
                      [](const NativeCell &cell) {
                        return cell.state == NativeState::Promoting ||
                               cell.state == NativeState::Completing ||
                               cell.state == NativeState::Submitted ||
                               cell.state == NativeState::Draining ||
                               cell.physical_handoff;
                      });
}

void Sliding::State::evidence(SlidingEvidence &result) const noexcept {
  result = SlidingEvidence{
      .status = has_failure ? status : Status::success(),
      .first_failure_terminal = first_failure_terminal,
      .terminal = terminal,
      .plan = plan,
      .first_failure = first_failure,
      .token = token,
      .generation = generation,
      .owner = owner,
      .planned = planned,
      .admitted = admitted,
      .terminal_frontier = terminal_frontier,
      .first_unsent = admitted,
      .fetch_calls = fetch_calls,
      .fetch_hits = fetch_hits,
      .promote_calls = promote_calls,
      .drain_calls = drain_calls,
      .persist_calls = persist_calls,
      .persist_issued = persist_issued,
      .persist_frontier = persist_frontier,
      .persist_completed_after_failure = persist_completed_after_failure,
      .fetch_bytes = fetch_bytes,
      .promote_bytes = promote_bytes,
      .drain_bytes = drain_bytes,
      .persist_bytes = persist_bytes,
      .persist_bytes_after_failure = persist_bytes_after_failure,
      .has_failure = has_failure,
      .first_failure_may_write = first_failure_may_write,
      .quarantined = quarantine,
  };
}

} // namespace rund::compute::detail::residency::execution

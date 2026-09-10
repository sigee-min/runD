#include "src/accel/kernel/terminal.hpp"

#include <atomic>
#include <cstdint>
#include <limits>
#include <thread>

namespace node_accel_contract {

bool NativeTerminalCellContract() {
  using rund::node::accel::detail::NativeTerminal;
  using rund::node::accel::detail::TerminalCell;
  using rund::node::accel::detail::TerminalState;

  TerminalCell cell{};
  if (cell.active() || cell.generation() != 0u ||
      cell.state() != TerminalState::Idle || !cell.arm(7u) || !cell.active() ||
      cell.generation() != 7u || cell.state() != TerminalState::Armed ||
      cell.arm(8u) || cell.resolve(8u, NativeTerminal::Known) ||
      !cell.resolve(7u, NativeTerminal::Known) ||
      cell.state() != TerminalState::Known ||
      cell.resolve(7u, NativeTerminal::UnknownMayWrite) ||
      cell.release(7u, NativeTerminal::UnknownMayWrite) ||
      !cell.release(7u, NativeTerminal::Known) || cell.active()) {
    return false;
  }

  if (!cell.arm(TerminalCell::MaxGeneration) ||
      !cell.resolve(TerminalCell::MaxGeneration,
                    NativeTerminal::UnknownMayWrite) ||
      cell.state() != TerminalState::UnknownMayWrite ||
      cell.release(TerminalCell::MaxGeneration, NativeTerminal::Known) ||
      !cell.release(TerminalCell::MaxGeneration,
                    NativeTerminal::UnknownMayWrite) ||
      cell.arm(0u) || cell.arm(std::numeric_limits<std::uint64_t>::max())) {
    return false;
  }

  TerminalCell raced{};
  std::atomic<unsigned> winners{};
  std::atomic<unsigned> winner{};
  if (!raced.arm(19u)) {
    return false;
  }
  std::thread known{[&] {
    if (raced.resolve(19u, NativeTerminal::Known)) {
      winners.fetch_add(1u, std::memory_order_relaxed);
      winner.store(1u, std::memory_order_relaxed);
    }
  }};
  std::thread unknown{[&] {
    if (raced.resolve(19u, NativeTerminal::UnknownMayWrite)) {
      winners.fetch_add(1u, std::memory_order_relaxed);
      winner.store(2u, std::memory_order_relaxed);
    }
  }};
  known.join();
  unknown.join();
  const unsigned resolved = winner.load(std::memory_order_relaxed);
  if (winners.load(std::memory_order_relaxed) != 1u || resolved == 0u ||
      !raced.release(19u, resolved == 1u ? NativeTerminal::Known
                                         : NativeTerminal::UnknownMayWrite) ||
      raced.active()) {
    return false;
  }
  return true;
}

} // namespace node_accel_contract

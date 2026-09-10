#pragma once

#include <atomic>
#include <cstdint>
#include <limits>

namespace rund::node::accel::detail {

// Backend-neutral native terminal certainty. Known permits native owner reuse;
// UnknownMayWrite is a terminal result for the host waiter only. It requires
// permanent quarantine of every owner the submitted commands may still touch.
enum class NativeTerminal : std::uint8_t {
  Known,
  UnknownMayWrite,
};

enum class TerminalState : std::uint8_t {
  Idle,
  Armed,
  Known,
  UnknownMayWrite,
};

// One allocation-free exact-generation terminal. The packed atomic makes the
// generation and state one compare/exchange authority: a late callback cannot
// terminalize a later generation and two terminal producers cannot both win.
class TerminalCell final {
public:
  [[nodiscard]] bool arm(const std::uint64_t generation) noexcept {
    if (generation == 0u || generation > MaxGeneration) {
      return false;
    }
    std::uint64_t expected = 0u;
    return word_.compare_exchange_strong(
        expected, pack(generation, TerminalState::Armed),
        std::memory_order_acq_rel, std::memory_order_acquire);
  }

  [[nodiscard]] bool resolve(const std::uint64_t generation,
                             const NativeTerminal terminal) noexcept {
    if (generation == 0u || generation > MaxGeneration) {
      return false;
    }
    std::uint64_t expected = pack(generation, TerminalState::Armed);
    const TerminalState resolved = terminal == NativeTerminal::Known
                                       ? TerminalState::Known
                                       : TerminalState::UnknownMayWrite;
    return word_.compare_exchange_strong(expected, pack(generation, resolved),
                                         std::memory_order_acq_rel,
                                         std::memory_order_acquire);
  }

  [[nodiscard]] bool release(const std::uint64_t generation,
                             const NativeTerminal terminal) noexcept {
    if (generation == 0u || generation > MaxGeneration) {
      return false;
    }
    const TerminalState resolved = terminal == NativeTerminal::Known
                                       ? TerminalState::Known
                                       : TerminalState::UnknownMayWrite;
    std::uint64_t expected = pack(generation, resolved);
    return word_.compare_exchange_strong(
        expected, 0u, std::memory_order_acq_rel, std::memory_order_acquire);
  }

  [[nodiscard]] std::uint64_t generation() const noexcept {
    return word_.load(std::memory_order_acquire) >> StateBits;
  }

  [[nodiscard]] TerminalState state() const noexcept {
    return static_cast<TerminalState>(word_.load(std::memory_order_acquire) &
                                      StateMask);
  }

  [[nodiscard]] bool active() const noexcept {
    return state() != TerminalState::Idle;
  }

  static constexpr std::uint64_t MaxGeneration =
      std::numeric_limits<std::uint64_t>::max() >> 2u;

private:
  static constexpr std::uint64_t StateBits = 2u;
  static constexpr std::uint64_t StateMask = (1u << StateBits) - 1u;

  [[nodiscard]] static constexpr std::uint64_t
  pack(const std::uint64_t generation, const TerminalState state) noexcept {
    return (generation << StateBits) | static_cast<std::uint64_t>(state);
  }

  std::atomic<std::uint64_t> word_{};
};

} // namespace rund::node::accel::detail

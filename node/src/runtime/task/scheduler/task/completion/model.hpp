#pragma once

#include <atomic>
#include <cstdint>

namespace rund::node {

struct CompletionLimits final {
  std::uint32_t capacity{};
  std::uint32_t result_bytes{256u};
  std::uint32_t result_alignment{64u};
};

struct CompletionLease final {
  void *authority{};
  std::uint32_t slot{};
  std::uint32_t generation{};

  [[nodiscard]] explicit operator bool() const noexcept {
    return authority != nullptr;
  }
};

struct CompletionSlot final {
  std::uint32_t slot{};
  std::uint32_t generation{};

  [[nodiscard]] explicit operator bool() const noexcept {
    return generation != 0u;
  }
};

enum class CompletionWaiterState : std::uint8_t {
  Idle,
  Linked,
  Waking,
};

struct CompletionWaiter final {
  void (*wake)(void *) noexcept {};
  void *value{};
  CompletionWaiter *next{};
  CompletionWaiter *previous{};
  std::atomic<CompletionWaiterState> state{CompletionWaiterState::Idle};
};

static_assert(sizeof(void *) != 8u || sizeof(CompletionWaiter) == 40u,
              "64-bit completion waiters must remain five machine words");

} // namespace rund::node

#pragma once

#include "../completion.hpp"

#include <algorithm>
#include <atomic>
#include <condition_variable>
#include <limits>
#include <memory>
#include <mutex>

namespace rund::node {

constexpr std::uint32_t kCompletionStripeCount = 64u;
constexpr std::uint32_t kCompletionPageSlots = 256u;
constexpr std::uint32_t kNoCompletionSlot =
    std::numeric_limits<std::uint32_t>::max();

enum class CompletionOutcome : std::uint8_t {
  Pending,
  Empty,
  Value,
  Failure,
};

[[nodiscard]] constexpr bool
CompletionTerminal(const task::Phase phase) noexcept {
  return phase == task::Phase::Completed || phase == task::Phase::Failed ||
         phase == task::Phase::Cancelled;
}

[[nodiscard]] constexpr ReasonCode
CompletionFailure(const ReasonCode code) noexcept {
  return code == ReasonCode::Ok ? ReasonCode::TaskFailed : code;
}

[[nodiscard]] constexpr task::Phase
CompletionFailurePhase(const ReasonCode code) noexcept {
  return code == ReasonCode::TaskCancelled ? task::Phase::Cancelled
                                           : task::Phase::Failed;
}

[[nodiscard]] constexpr bool CompletionAllowed(const task::Phase from,
                                               const task::Phase to) noexcept {
  switch (from) {
  case task::Phase::Idle:
    return to == task::Phase::Admitted;
  case task::Phase::Admitted:
    return to == task::Phase::Ready;
  case task::Phase::Ready:
    return to == task::Phase::Running;
  case task::Phase::Running:
    return to == task::Phase::Parked || to == task::Phase::Committing;
  case task::Phase::Parked:
    return to == task::Phase::Ready;
  case task::Phase::Committing:
    return CompletionTerminal(to);
  case task::Phase::Completed:
  case task::Phase::Failed:
  case task::Phase::Cancelled:
    return false;
  }
  return false;
}

[[nodiscard]] CompletionWaiter *
OrderCompletionWaiters(CompletionWaiter *waiter) noexcept;
void WakeCompletionWaiters(CompletionWaiter *ordered) noexcept;

struct CompletionPool::Store final {
  struct Stripe final {
    std::mutex mutex{};
    std::condition_variable ready{};
  };

  struct Cell final {
    union Link {
      CompletionWaiter *wait_head;
      std::uint32_t free_next;

      constexpr Link() noexcept : wait_head(nullptr) {}
    } link{};
    void *result{};
    std::uint32_t generation{1u};
    std::uint32_t observers{};
    ReasonCode code{ReasonCode::Ok};
    task::Phase phase{task::Phase::Idle};
    CompletionOutcome outcome{CompletionOutcome::Pending};
    bool producer_live{};
  };

  struct ResultHeader final {
    const void *type{};
    DestroyFn destroy{};
  };

  static_assert(sizeof(void *) != 8u || sizeof(Cell) == 32u,
                "64-bit completion cells must remain half a cache line");

  ~Store();

  [[nodiscard]] void *result(std::uint32_t slot) const noexcept;
  [[nodiscard]] Cell &at(std::uint32_t slot) const noexcept;
  [[nodiscard]] bool contains(std::uint32_t slot) const noexcept;
  [[nodiscard]] bool grow() noexcept;
  [[nodiscard]] ResultHeader &result_header(Cell &cell) const noexcept;
  [[nodiscard]] Stripe &stripe(std::uint32_t slot) const noexcept;
  void destroy_value(std::uint32_t slot, Cell &cell) noexcept;
  void recycle(std::uint32_t slot, Cell &cell) noexcept;

  std::atomic<std::uint64_t> refs{1u};
  std::mutex mutex{};
  std::unique_ptr<Stripe[]> stripes{};
  std::uint32_t stripe_count{};
  std::unique_ptr<std::atomic<Cell *>[]> pages{};
  std::uint32_t page_count{};
  std::atomic<std::uint32_t> created{0u};
  std::uint32_t free_head{kNoCompletionSlot};
  CompletionLimits limits{};
  bool retired{};
};

} // namespace rund::node

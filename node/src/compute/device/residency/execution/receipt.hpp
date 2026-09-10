#pragma once

#include "plan.hpp"

#include <condition_variable>
#include <cstdint>
#include <mutex>

namespace rund::compute::detail::residency::execution {

// One accelerator run-level transport receipt. Authority owns the exact Node
// journal and ordering. Receipt owns only the one native submission boundary,
// final callback, and wait lifetime; it cannot issue or reorder an epoch.
class Receipt final {
public:
  Receipt() = default;
  Receipt(const Receipt &) = delete;
  Receipt &operator=(const Receipt &) = delete;

  [[nodiscard]] bool arm(const Plan &, std::uint64_t token,
                         std::uint64_t generation,
                         std::uint64_t started_ns) noexcept;
  [[nodiscard]] bool submit() noexcept;
  [[nodiscard]] bool terminal(const Evidence &) noexcept;
  [[nodiscard]] ReceiptSnapshot wait() noexcept;

private:
  enum class State : std::uint8_t { Idle, Armed, Submitted, Ready };

  std::mutex gate_;
  std::condition_variable ready_;
  ReceiptSnapshot snapshot_{};
  State state_{State::Idle};
};

} // namespace rund::compute::detail::residency::execution

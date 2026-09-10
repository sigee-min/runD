#pragma once

#include <cstdint>

namespace rund::compute::detail::residency {

namespace execution {
class Sliding;
} // namespace execution

class Authority;
class SlidingOwner;

class ExecutionSlidingFinal final {
public:
  ExecutionSlidingFinal() = default;
  ExecutionSlidingFinal(const ExecutionSlidingFinal &) = delete;
  ExecutionSlidingFinal &operator=(const ExecutionSlidingFinal &) = delete;
  ExecutionSlidingFinal(ExecutionSlidingFinal &&) noexcept = default;
  ExecutionSlidingFinal &operator=(ExecutionSlidingFinal &&) noexcept = default;
  [[nodiscard]] explicit operator bool() const noexcept { return nonce_ != 0u; }

private:
  friend class Authority;
  friend class SlidingOwner;
  std::uint64_t plan_{};
  std::uint64_t token_{};
  std::uint64_t generation_{};
  std::uint64_t owner_{};
  std::uint64_t sliding_nonce_{};
  std::uint64_t nonce_{};
  bool success_{};
};

class ExecutionSlidingReceipt final {
public:
  ExecutionSlidingReceipt() = default;
  ExecutionSlidingReceipt(const ExecutionSlidingReceipt &) = delete;
  ExecutionSlidingReceipt &operator=(const ExecutionSlidingReceipt &) = delete;
  ExecutionSlidingReceipt(ExecutionSlidingReceipt &&) noexcept = default;
  ExecutionSlidingReceipt &
  operator=(ExecutionSlidingReceipt &&) noexcept = default;
  [[nodiscard]] explicit operator bool() const noexcept { return nonce_ != 0u; }

private:
  friend class Authority;
  friend class SlidingOwner;
  friend class execution::Sliding;
  std::uint64_t plan_{};
  std::uint64_t token_{};
  std::uint64_t generation_{};
  std::uint64_t owner_{};
  std::uint64_t sliding_nonce_{};
  std::uint64_t nonce_{};
  bool success_{};
};

using ExecutionSlidingPublication = void (*)(void *) noexcept;

enum class FinalAbort : std::uint8_t {
  Invalid,
  Closed,
  Quarantined,
};

enum class SlidingFinalPhase : std::uint8_t {
  None,
  Inflight,
  Quarantined,
};

} // namespace rund::compute::detail::residency

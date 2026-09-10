#pragma once

#include "../../execution/evidence.hpp"
#include "../transfer.hpp"
#include "result.hpp"

#include <array>
#include <cstddef>
#include <cstdint>
#include <span>

namespace rund::compute::detail::residency {

class Authority;

struct ExecutionTicket final {
  std::uint64_t token{};
  std::uint64_t generation{};
  std::uint64_t plan{};
  std::uint64_t epoch{};
  std::uint64_t sequence{};
  std::uint8_t phase{};
  std::uint8_t bank{};
  std::span<const CacheBinding> bindings;
  std::span<const CacheTransition> transitions;
  std::uint32_t backing_mask{};
  std::uint32_t transfer_mask{};
  std::uint32_t coherent_mask{};
  bool may_write{};
};

struct ExecutionLease final {
  AuthorityFailure failure{AuthorityFailure::Invalid};
  std::uint64_t token{};
  std::uint64_t generation{};
  std::uint64_t owner_nonce{};
  std::uint64_t plan{};
  std::uint64_t epochs{};
  [[nodiscard]] explicit operator bool() const noexcept {
    return failure == AuthorityFailure::None;
  }
};

enum class ExecutionTerminal : std::uint8_t {
  Success,
  Failure,
  UnknownMayWrite,
};

struct ExecutionFailure final {
  std::uint64_t epoch{};
  std::uint8_t phases{};
  std::uint8_t may_write{};
  std::uint8_t terminal{};
};

using ExecutionProgress = execution::Progress;

struct ExecutionClose final {
  static constexpr std::size_t FailureCapacity = 6u;

  AuthorityFailure failure{AuthorityFailure::Invalid};
  ExecutionProgress progress{};
  std::array<ExecutionFailure, FailureCapacity> failures{};
  std::size_t failure_count{};
  bool success{};
  bool quarantined{};
  RegistrationResult registration{RegistrationResult::Invalid};

  [[nodiscard]] explicit operator bool() const noexcept {
    return failure == AuthorityFailure::None;
  }
};

} // namespace rund::compute::detail::residency

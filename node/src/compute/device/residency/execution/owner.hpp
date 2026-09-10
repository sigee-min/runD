#pragma once

#include "receipt.hpp"

#include <array>
#include <atomic>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <memory>

namespace rund::compute::detail::residency::execution {

// Cold backend owner for one already-accounted prepared Pipeline. Run-specific
// Plan identity is deliberately not retained here: a fixed owner can be bound
// to successive invocation Plans without allocating on the warm path.
struct Owner final {
  std::shared_ptr<void> native{};
  std::uint64_t host_bytes{};
  std::uint64_t device_bytes{};
  std::uint64_t staging_bytes{};

  [[nodiscard]] explicit operator bool() const noexcept {
    return native != nullptr;
  }
  [[nodiscard]] bool retained_bytes(std::uint64_t &result) const noexcept {
    result = 0u;
    if (host_bytes > std::numeric_limits<std::uint64_t>::max() - device_bytes) {
      return false;
    }
    result = host_bytes + device_bytes;
    if (result > std::numeric_limits<std::uint64_t>::max() - staging_bytes) {
      result = 0u;
      return false;
    }
    result += staging_bytes;
    return true;
  }
};

using Completion = void (*)(void *, NativeEvidence &&) noexcept;

// Caller-owned fixed warm slot. The selected-local array remains alive through
// asynchronous prepared-Pipeline evidence projection. The adapter fills only
// NativeEvidence and invokes one completion; it cannot claim Host services.
struct Control final {
  std::uint64_t token{};
  std::uint64_t generation{};
  NativeEvidence *evidence{};
  std::array<std::uint32_t, UseCapacity> locals{};
  std::size_t local_count{};
  Completion completion{};
  void *user{};

  // Adapter-bound snapshot. `active` is the exact async lifetime gate: run
  // inputs above may be prepared for a later invocation only after terminal
  // releases this immutable snapshot.
  std::atomic_bool active{false};
  std::uint64_t bound_plan{};
  std::uint64_t bound_token{};
  std::uint64_t bound_generation{};
  NativeEvidence *bound_evidence{};
  Completion bound_completion{};
  void *bound_user{};

  [[nodiscard]] explicit operator bool() const noexcept {
    return token != 0u && generation != 0u && evidence != nullptr &&
           completion != nullptr;
  }
};

} // namespace rund::compute::detail::residency::execution

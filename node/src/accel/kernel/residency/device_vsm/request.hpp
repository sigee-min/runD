#pragma once

#include "evidence.hpp"
#include "proof.hpp"

#include <atomic>
#include <cstdint>
#include <memory>

namespace rund::node::accel::detail {

class DeviceVsmSubmissionControl final {
public:
  void reset() noexcept { count_value.store(0u, std::memory_order_release); }

  [[nodiscard]] bool accept() noexcept {
    std::uint32_t expected = 0u;
    return count_value.compare_exchange_strong(
        expected, 1u, std::memory_order_acq_rel, std::memory_order_acquire);
  }

  [[nodiscard]] std::uint32_t count() const noexcept {
    return count_value.load(std::memory_order_acquire);
  }

private:
  std::atomic_uint32_t count_value{};
};

// Aggregate-only device-owned VSM handoff. Host page service, projection,
// release, returned, and wake function pointers are intentionally absent.
struct DeviceVsmRequest final {
  std::shared_ptr<const DeviceVsmProof> proof{};
  std::shared_ptr<void> lowering{};
  std::shared_ptr<const void> admission{};
  std::uint64_t token{};
  std::uint64_t generation{};
  std::uint64_t nonce{};
  DeviceVsmSubmissionControl *submission_control{};
  DeviceVsmFinalCompletion final{};
  void *user{};
};

} // namespace rund::node::accel::detail

#pragma once

#include <atomic>
#include <cstdint>

namespace rund::node::accel::detail {

enum class SubmitKind : std::uint8_t {
  None,
  Work,
  Transfer,
};

class DeviceLossFault final {
public:
  DeviceLossFault() = default;
  DeviceLossFault(const DeviceLossFault &) = delete;
  DeviceLossFault &operator=(const DeviceLossFault &) = delete;

  [[nodiscard]] bool arm(const SubmitKind kind) noexcept {
    if (kind == SubmitKind::None) {
      return false;
    }
    SubmitKind expected = SubmitKind::None;
    return pending_.compare_exchange_strong(
        expected, kind, std::memory_order_acq_rel,
        std::memory_order_acquire);
  }

  [[nodiscard]] bool take(const SubmitKind kind) noexcept {
    if (kind == SubmitKind::None) {
      return false;
    }
    SubmitKind expected = kind;
    return pending_.compare_exchange_strong(
        expected, SubmitKind::None, std::memory_order_acq_rel,
        std::memory_order_acquire);
  }

private:
  std::atomic<SubmitKind> pending_{SubmitKind::None};
};

} // namespace rund::node::accel::detail

#pragma once

#include <accel/check.hpp>
#include <accel/context/evidence.hpp>
#include <accel/device.hpp>
#include <kernel/program/compute/model.hpp>

#include <cstdint>
#include <memory>

namespace rund {

struct AccelContext {
  AccelCheck check{false, "accel_context_invalid"};
  std::uint64_t id = 0u;
  AccelDevice pick{};
  AccelApi api = AccelApi::Auto;
  rund::kernel::ComputeCaps caps{};
  std::shared_ptr<void> owner{};
  AccelContextEvidence evidence{};

  [[nodiscard]] constexpr explicit operator bool() const noexcept {
    return check.ok;
  }
};

}  // namespace rund
